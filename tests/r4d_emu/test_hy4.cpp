/* tests/r4d_emu: Hy4-flash's small ops (r4d_ihc_bf16.hip, r4d_engram_bf16.hip, r4d_mla_kv_bf16.hip,
 * r4d_dfly.hip)
 * on the host SIMT emulator, each held to libref's op (libref/ref_hy4.cpp) on the same bytes.
 * ihc_post, mla_kv_split and Engram's ids, mask and window compare byte for byte; the others within
 * the bound their kernels state (one bf16 ulp of a differently ordered or fused sum). Row padding of
 * a pitched operand must come back untouched. Needs libref (RADIANCE_HOME), as test_kda. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "../hy4_dfly_cases.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <vector>

static const RadKernelInfo* libref_op(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
        if (open && open() < 0) return nullptr;
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i)
        if (at(i)->op && !std::strcmp(at(i)->op, op)) return at(i);
    return nullptr;
}

static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: return 4;
        case RAD_BF16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

struct Opd {
    uint32_t dt = RAD_BF16;
    std::vector<int64_t> shape;
    int64_t pitch = 0;
    std::vector<uint8_t> bytes;
    bool absent = false, out = false;
    double rtol = 0, atol = 0;
    int64_t elements() const {
        if (shape.size() == 2 && pitch) return shape[0] * pitch;
        int64_t n = 1;
        for (int64_t e : shape) n *= e;
        return n;
    }
    RadTensor tensor(void* data) const {
        RadTensor t{};
        t.data = absent ? nullptr : data;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[(size_t)i];
            t.stride[i] = st;
            st *= shape[(size_t)i];
        }
        if (shape.size() == 2 && pitch) t.stride[0] = pitch;
        return t;
    }
};
static Opd opd(uint32_t dt, std::vector<int64_t> shape, int64_t pitch = 0) {
    Opd o;
    o.dt = dt;
    o.shape = std::move(shape);
    o.pitch = pitch;
    o.bytes.assign((size_t)o.elements() * esz(dt), 0x7e);
    return o;
}
static Opd absent() { Opd o; o.absent = true; return o; }

static std::vector<int64_t> offsets(const RadTensor& t) {
    std::vector<int64_t> o;
    const int64_t d0 = t.shape[0], d1 = t.rank > 1 ? t.shape[1] : 1, d2 = t.rank > 2 ? t.shape[2] : 1;
    for (int64_t i = 0; i < d0; ++i)
        for (int64_t j = 0; j < d1; ++j)
            for (int64_t k = 0; k < d2; ++k)
                o.push_back(i * t.stride[0] + (t.rank > 1 ? j * t.stride[1] : 0) +
                            (t.rank > 2 ? k * t.stride[2] : 0));
    return o;
}
static void fill(Opd& o, const std::function<double(int64_t)>& f) {
    const RadTensor t = o.tensor(o.bytes.data());
    int64_t idx = 0;
    for (int64_t off : offsets(t)) {
        const double v = f(idx++);
        switch (o.dt) {
            case RAD_BF16: ((uint16_t*)o.bytes.data())[off] = f2b((float)v); break;
            case RAD_F32: ((float*)o.bytes.data())[off] = (float)v; break;
            case RAD_I32: ((int32_t*)o.bytes.data())[off] = (int32_t)v; break;
            case RAD_I64: ((int64_t*)o.bytes.data())[off] = (int64_t)v; break;
            default: break;
        }
    }
}
static double get(uint32_t dt, const uint8_t* p, int64_t off) {
    switch (dt) {
        case RAD_BF16: return b2f(((const uint16_t*)p)[off]);
        case RAD_F32: return ((const float*)p)[off];
        case RAD_I32: return ((const int32_t*)p)[off];
        case RAD_I64: return (double)((const int64_t*)p)[off];
        default: return ((const uint8_t*)p)[off];
    }
}

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float u() { return (float)(next() >> 8) / 8388608.0f - 1.0f; }
    float n() { return (u() + u() + u() + u()) * 0.866f; }
    uint32_t r(uint32_t m) { return next() % m; }
};

static RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

static int g_fails = 0, g_cases = 0;
/* g_quiet: no line for a passing case. g_ref_shape: libref's RAD_E_SHAPE still compares (dfly_sel's
 * failed rows, which a debug libref also reports; the outputs are written either way). */
static bool g_quiet = false, g_ref_shape = false;

static void check(const char* op, RadLaunchFn kernel, const std::string& what, std::vector<RadParam> p,
                  std::vector<Opd>& ops) {
    ++g_cases;
    const RadKernelInfo* ref = libref_op(op);
    if (!ref) {
        std::printf("FAIL %s: libref's %s not loadable (set RADIANCE_HOME)\n", what.c_str(), op);
        ++g_fails;
        return;
    }
    const size_t n = ops.size();
    std::vector<std::vector<uint8_t>> a(n), b(n);
    std::vector<RadTensor> ta(n), tb(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = b[i] = ops[i].bytes;
        ta[i] = ops[i].tensor(a[i].data());
        tb[i] = ops[i].tensor(b[i].data());
    }
    RadArgs ar{}, ak{};
    ar.t = ta.data(); ar.n_t = (int)n; ar.p = p.data(); ar.n_p = (int)p.size(); ar.world_size = 1;
    ak.t = tb.data(); ak.n_t = (int)n; ak.p = p.data(); ak.n_p = (int)p.size(); ak.world_size = 1;
    const int rr = ref->launch(&ar, nullptr), rk = kernel(&ak, nullptr);
    if ((rr != RAD_OK && !(g_ref_shape && rr == RAD_E_SHAPE)) || rk != RAD_OK) {
        std::printf("FAIL %s: libref rc=%d kernel rc=%d\n", what.c_str(), rr, rk);
        ++g_fails;
        return;
    }
    int64_t bad = 0, total = 0;
    bool padding = true;
    for (size_t i = 0; i < n; ++i) {
        if (ops[i].absent) continue;
        std::vector<char> live(a[i].size() / esz(ops[i].dt), 0);
        for (int64_t off : offsets(ta[i])) {
            live[(size_t)off] = 1;
            if (!ops[i].out) continue;
            const double e = get(ops[i].dt, a[i].data(), off), g = get(ops[i].dt, b[i].data(), off);
            ++total;
            if (!std::isfinite(g) || std::fabs(g - e) > ops[i].atol + ops[i].rtol * std::fabs(e)) {
                if (!bad) std::printf("  first: operand %zu element %lld got %.9g want %.9g\n", i, (long long)off, g, e);
                ++bad;
            }
        }
        const size_t es = esz(ops[i].dt);
        for (size_t off = 0; off < live.size(); ++off)
            if (!live[off] && std::memcmp(&b[i][off * es], &ops[i].bytes[off * es], es)) padding = false;
        if (!ops[i].out && std::memcmp(b[i].data(), ops[i].bytes.data(), b[i].size())) padding = false;
    }
    if (bad || !padding || !total) {
        std::printf("FAIL %s: %lld of %lld out of bound%s\n", what.c_str(), (long long)bad,
                    (long long)total, padding ? "" : ", padding or an input written");
        ++g_fails;
    } else {
        if (!g_quiet) std::printf("ok   %s (%lld values)\n", what.c_str(), (long long)total);
    }
}

static const double kUlp = 0.0079;   /* one bf16 ulp, relative */

static void ihc_pre() {
    struct G { int64_t hc, n, M, pad; bool post; };
    for (const G& g : { G{ 4, 96, 3, 0, true }, G{ 4, 40, 2, 5, true }, G{ 2, 17, 3, 0, true },
                        G{ 3, 64, 2, 0, false }, G{ 8, 24, 2, 3, true } }) {
        Rng rng((uint32_t)(g.hc * 100 + g.n));
        const int64_t W = g.hc * g.n, rows = g.post ? 2 * g.hc : g.hc;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 2; });
        ops.push_back(opd(RAD_F32, { rows, W }));
        fill(ops[1], [&](int64_t) { return rng.n() * 0.05; });
        ops.push_back(opd(RAD_F32, { g.post ? 2 : 1 }));
        fill(ops[2], [&](int64_t i) { return i ? 2.1 : 3.0; });
        ops.push_back(opd(RAD_F32, { rows }));
        fill(ops[3], [&](int64_t) { return rng.n() * 0.7; });
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        ops[4].out = true; ops[4].rtol = kUlp; ops[4].atol = 1e-5;
        if (g.post) {
            ops.push_back(opd(RAD_F32, { g.M, g.hc }, g.pad ? g.hc + 2 : 0));
            ops[5].out = true; ops[5].rtol = 2e-4; ops[5].atol = 1e-6;
        } else {
            ops.push_back(absent());
        }
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pi("rows", rows),
                                    pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
        if (g.post) p.push_back(pf("post_mag", 2.0));
        char w[120];
        std::snprintf(w, sizeof w, "ihc_pre hc %lld n %lld M %lld%s%s", (long long)g.hc, (long long)g.n,
                      (long long)g.M, g.post ? "" : " head", g.pad ? " pitched" : "");
        check("ihc_pre", r4d_rad_ihc_pre, w, p, ops);
    }
}

static void ihc_post() {
    struct G { int64_t hc, n, M, pad; };
    for (const G& g : { G{ 4, 64, 3, 0 }, G{ 4, 64, 2, 8 }, G{ 2, 17, 3, 0 }, G{ 3, 24, 2, 5 } }) {
        Rng rng((uint32_t)(g.hc * 10 + g.n));
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 3; });
        ops.push_back(opd(RAD_F32, { g.M, g.hc }));
        fill(ops[1], [&](int64_t) { return 0.5 + rng.u(); });
        ops.push_back(opd(RAD_BF16, { g.M, g.hc * g.n }, g.pad ? g.hc * g.n + g.pad : 0));
        fill(ops[2], [&](int64_t) { return rng.n() * 2; });
        ops[2].out = true;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), ps("dtype", "bf16") };
        char w[120];
        std::snprintf(w, sizeof w, "ihc_post hc %lld n %lld M %lld%s, bit for bit", (long long)g.hc,
                      (long long)g.n, (long long)g.M, g.pad ? " pitched" : "");
        check("ihc_post", r4d_rad_ihc_post, w, p, ops);
    }
}

static void gate_mul() {
    struct G { int64_t M, n, pad; };
    for (const G& g : { G{ 3, 64, 0 }, G{ 2, 80, 8 }, G{ 3, 13, 0 }, G{ 2, 37, 3 } }) {
        Rng rng((uint32_t)(g.n * 3 + g.M));
        std::vector<Opd> ops;
        for (int i = 0; i < 2; ++i) {
            ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
            fill(ops.back(), [&](int64_t) { return rng.n() * (i ? 4 : 2); });
        }
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 1e-30;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "gate_mul M %lld n %lld%s", (long long)g.M, (long long)g.n, g.pad ? " pitched" : "");
        check("gate_mul", r4d_rad_gate_mul, w, p, ops);
    }
}

static void mla_kv() {
    for (const char* mode : { "absorb", "expand" })
        for (int64_t M : { 1, 3, 9 }) {
            const bool ab = !std::strcmp(mode, "absorb");
            const int64_t H = 2, kd = 192, vd = 256, L = 512, xw = ab ? kd : L, yw = ab ? L : vd;
            Rng rng((uint32_t)(M + (ab ? 7 : 70)));
            std::vector<Opd> ops;
            ops.push_back(opd(RAD_BF16, { M, H, xw }));
            fill(ops[0], [&](int64_t) { return rng.n(); });
            ops.push_back(opd(RAD_BF16, { H * (kd + vd), L }));
            fill(ops[1], [&](int64_t) { return rng.n() * 0.05; });
            ops.push_back(opd(RAD_BF16, { M, H, yw }));
            ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 2e-5;
            std::vector<RadParam> p = { pi("M", M), pi("heads", H), pi("k_dim", kd), pi("v_dim", vd),
                                        pi("latent_dim", L), ps("mode", mode), ps("dtype", "bf16") };
            char w[100];
            std::snprintf(w, sizeof w, "mla_project_kv %s heads %lld M %lld", mode, (long long)H, (long long)M);
            check("mla_project_kv", r4d_rad_mla_project_kv, w, p, ops);
        }
    struct G { int64_t M, H, kd, vd, rd, pad; };
    for (const G& g : { G{ 3, 2, 16, 24, 8, 0 }, G{ 2, 3, 16, 8, 8, 8 }, G{ 3, 2, 13, 10, 6, 0 }, G{ 2, 2, 12, 9, 5, 3 } }) {
        Rng rng((uint32_t)(g.M * 7 + g.kd));
        const int64_t bw = g.H * (g.kd + g.vd), kw = g.H * (g.kd + g.rd), vw = g.H * g.vd;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, bw }, g.pad ? bw + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_BF16, { g.M, g.rd }, g.pad ? g.rd + g.pad : 0));
        fill(ops[1], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_BF16, { g.M, kw }, g.pad ? kw + g.pad : 0));
        ops.push_back(opd(RAD_BF16, { g.M, vw }, g.pad ? vw + g.pad : 0));
        ops[2].out = ops[3].out = true;
        std::vector<RadParam> p = { pi("M", g.M), pi("heads", g.H), pi("k_dim", g.kd), pi("v_dim", g.vd),
                                    pi("rope_dim", g.rd), ps("dtype", "bf16") };
        char w[120];
        std::snprintf(w, sizeof w, "mla_kv_split %lld/%lld/%lld heads %lld%s, bit for bit", (long long)g.kd,
                      (long long)g.vd, (long long)g.rd, (long long)g.H, g.pad ? " pitched" : "");
        check("mla_kv_split", r4d_rad_mla_kv_split, w, p, ops);
    }
}

static void engram_ids() {
    const int64_t heads = 4, ngram = 3, pad = 90, msize = 100, image = 95, video = 96;
    /* Form 4: a prefill whose last sequence hashes three tokens of its following prompt into
     * `ahead_ids` (the read-ahead), `tok` running on past the step's rows. */
    for (int form = 0; form < 5; ++form) {
        const bool decode = form & 1, wide = form & 2, ahd = form == 4;
        const int64_t A = ahd ? 3 : 0;
        const int64_t S = 3, W = decode ? 4 : 2;
        const std::vector<int64_t> Ts = decode ? std::vector<int64_t>{ 3, 1, 2 } : std::vector<int64_t>{ 4, 1, 3 };
        int64_t M = 0;
        for (int64_t t : Ts) M += t;
        Rng rng((uint32_t)(form + 3));
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_I32, { M + A }));
        fill(ops[0], [&](int64_t) {
            const uint32_t r = rng.r(12);
            return r == 0 ? (double)pad : r == 1 ? (double)image : r == 2 ? (double)video : r == 3 ? 150.0 : (double)rng.r(90);
        });
        ops.push_back(opd(RAD_I32, { 4, W }));
        fill(ops[1], [&](int64_t) { return (double)rng.r(90); });
        ops[1].out = true;
        std::vector<int> cu = { 0 };
        for (int64_t t : Ts) cu.push_back(cu.back() + (int)t);
        ops.push_back(opd(RAD_I32, { S + 1 }));
        fill(ops[2], [&](int64_t i) { return cu[(size_t)i]; });
        ops.push_back(opd(RAD_I32, { M }));
        {
            int64_t i = 0;
            for (int64_t s = 0; s < S; ++s)
                for (int64_t t = 0; t < Ts[(size_t)s]; ++t, ++i)
                    ((int32_t*)ops[3].bytes.data())[i] = (int32_t)((s == 0 ? 0 : 500) + t);
        }
        ops.push_back(opd(RAD_I32, { msize }));
        fill(ops[4], [&](int64_t) { return (double)rng.r(60); });
        ops.push_back(opd(RAD_I64, { ngram }));
        { const int64_t m[3] = { 1000003, 7919000117ll, 0x5851F42D4C957F2Dll };
          for (int i = 0; i < 3; ++i) ((int64_t*)ops[5].bytes.data())[i] = m[i]; }
        ops.push_back(opd(RAD_I64, { heads }));
        fill(ops[6], [&](int64_t i) { return (double)(1009 + 13 * i); });
        ops.push_back(opd(RAD_I64, { heads }));
        fill(ops[7], [&](int64_t i) { return (double)(i * 2003); });
        if (decode) {
            ops.push_back(opd(RAD_I32, { S }));
            fill(ops[8], [&](int64_t i) { return (double)(1 + i % 3); });
        } else {
            ops.push_back(absent());
        }
        ops.push_back(opd(RAD_I32, { S }));
        { const int32_t slots[3] = { 3, -1, 0 };
          fill(ops[9], [&](int64_t i) { return (double)slots[i]; }); }
        ops.push_back(opd(wide ? RAD_I64 : RAD_I32, { M, heads }));
        ops.push_back(opd(RAD_I32, { M }));
        ops[10].out = ops[11].out = true;
        if (ahd) {
            ops.push_back(opd(RAD_I32, { A, heads }));
            ops[12].out = true;
        }
        std::vector<RadParam> p = { pi("M", M), pi("heads", heads), pi("ngram", ngram), pi("pad", pad),
                                    pi("map_size", msize), pi("image", image), pi("video", video) };
        char w[100];
        std::snprintf(w, sizeof w, "engram_ids %s%s, ids i%d, bit for bit", decode ? "decode" : "prefill",
                      ahd ? " with ahead_ids" : "", wide ? 64 : 32);
        check("engram_ids", r4d_rad_engram_ids, w, p, ops);
    }
}

static void engram_norm_gate() {
    struct N { int64_t M, heads, n, pad; double eps; };
    for (const N& g : { N{ 3, 4, 24, 0, 0.0078125 }, N{ 2, 3, 37, 5, 1.19e-7 } }) {
        Rng rng((uint32_t)(g.n));
        const int64_t W = g.heads * g.n;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t) { return rng.n() * 3; });
        ops.push_back(opd(RAD_BF16, { W }));
        fill(ops[1], [&](int64_t) { return 1 + rng.n() * 0.3; });
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        ops[2].out = true; ops[2].rtol = kUlp; ops[2].atol = 1e-30;
        std::vector<RadParam> p = { pi("M", g.M), pi("heads", g.heads), pi("n", g.n), pf("eps", g.eps),
                                    ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "engram_norm heads %lld n %lld eps %g", (long long)g.heads, (long long)g.n, g.eps);
        check("engram_norm", r4d_rad_engram_norm, w, p, ops);
    }
    struct G { int64_t M, hc, n, pad; double eps; };
    for (const G& g : { G{ 4, 2, 48, 0, 0.0078125 }, G{ 3, 4, 20, 6, 1.19e-7 }, G{ 3, 1, 33, 0, 1e-6 } }) {
        Rng rng((uint32_t)(g.n * 5));
        const int64_t W = g.hc * g.n;
        std::vector<Opd> ops;
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[0], [&](int64_t i) { return i / W == 1 ? 0.0 : rng.n() * 2; });   /* row 1: d == 0 */
        ops.push_back(opd(RAD_BF16, { g.M, g.n }, g.pad ? g.n + g.pad : 0));
        fill(ops[1], [&](int64_t) { return rng.n(); });
        ops.push_back(opd(RAD_I32, { g.M }));
        fill(ops[2], [&](int64_t i) { return i % 3 == 2 ? 0.0 : 1.0; });
        for (int k = 0; k < 4; ++k) {
            ops.push_back(opd(RAD_BF16, { k == 2 ? g.n : W }));
            fill(ops.back(), [&](int64_t) { return 1 + rng.n() * 0.2; });
        }
        ops.push_back(opd(RAD_BF16, { g.M, W }, g.pad ? W + g.pad : 0));
        fill(ops[7], [&](int64_t) { return rng.n() * 2; });
        ops[7].out = true; ops[7].rtol = 2 * kUlp; ops[7].atol = 0.016;
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pf("eps", g.eps), ps("dtype", "bf16") };
        char w[100];
        std::snprintf(w, sizeof w, "engram_gate hc %lld n %lld eps %g%s", (long long)g.hc, (long long)g.n, g.eps,
                      g.pad ? " pitched" : "");
        check("engram_gate", r4d_rad_engram_gate, w, p, ops);
    }
}

/* dfly_sel (r4d_dfly.hip): every golden case alone and all of one G in one call, then 1000 seeded
 * random geometries (sources -1 and past the snapshot, unheld rows, empty sequences, wide tables);
 * then 100 with tables past `blocks` (the rows fail to -1, len 0). Byte for byte, with
 * and without `count`; a pitched `out` keeps its padding and its columns past `blocks`. */
static void dfly_sel_case(const hy4dfly::Case& c, bool with_count, int64_t out_pad, const char* what) {
    std::vector<Opd> ops;
    auto put = [&](std::vector<int64_t> shape, const std::vector<int64_t>& v) {
        ops.push_back(opd(RAD_I32, std::move(shape)));
        fill(ops.back(), [&](int64_t i) { return (double)v[(size_t)i]; });
    };
    put({ c.R, c.G }, c.sel);
    put({ c.R }, c.nsel);
    put({ c.S }, c.src);
    put({ c.S, c.pitch }, c.bt);
    put({ c.M }, c.pos);
    put({ c.S + 1 }, c.cu);
    ops.push_back(opd(RAD_I32, { c.M, c.blocks }, out_pad ? c.blocks + out_pad : 0));
    ops.push_back(opd(RAD_I32, { c.M }));
    ops.push_back(with_count ? opd(RAD_I32, { c.M }) : absent());
    ops[6].out = ops[7].out = ops[8].out = true;
    std::vector<RadParam> p = { pi("M", c.M), pi("blocks_in", c.G), pi("blocks", c.blocks),
                                pi("ratio", c.ratio), pi("seqs", c.S) };
    check("dfly_sel", r4d_rad_dfly_sel, what, p, ops);
}

static void dfly_sel() {
    const int before = g_fails, cases = g_cases;
    g_quiet = true;
    for (int i = 0; i < kDflyCases; ++i)
        dfly_sel_case(hy4dfly::golden({ i }, 1000 + (uint64_t)i), i & 1, (i % 3) == 0 ? 3 : 0,
                      "dfly_sel golden case, one sequence");
    for (const int64_t G : { 4, 512 })
        dfly_sel_case(hy4dfly::golden(hy4dfly::golden_of_g(G), 77), true, 0,
                      G == 4 ? "dfly_sel every G 4 golden case in one call"
                             : "dfly_sel every G 512 golden case in one call");
    for (uint64_t seed = 1; seed <= 1000; ++seed)
        dfly_sel_case(hy4dfly::random(seed), seed & 1, (seed % 4) == 0 ? 2 : 0, "dfly_sel random geometry");
    g_ref_shape = true;
    for (uint64_t seed = 1; seed <= 100; ++seed)
        dfly_sel_case(hy4dfly::random(5000 + seed, true), true, 0, "dfly_sel random geometry, tables past blocks");
    g_ref_shape = false;
    g_quiet = false;
    std::printf("dfly_sel: %d case(s), %d failure(s)\n", g_cases - cases, g_fails - before);
}

/* ---- ihc_pre's order (r4d_ihc.h), restated on the host: 512 threads own the row's 8-value pieces
 * t, t + 512, ...; per thread the pieces and their values ascending; a wave's lanes by the xor
 * butterfly 16, 8, 4, 2, 1; the 16 waves from 0.0f in order. The kernel must give these bytes, on
 * the 16-byte form and on the scalar-load form alike. */
static float host_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static void host_ihc_pre_row(const uint16_t* h, const float* fn, int64_t fp, const float* scale,
                             const float* base, int hc, int n, bool post, float eps, float heps,
                             float pmag, uint16_t* x, float* prow) {
    const int R = post ? 2 * hc : hc, W = hc * n, T = 512;
    std::vector<float> acc((size_t)T * (R + 1), 0.0f);
    for (int p = 0; 8 * p < W; ++p) {
        const int t = p % T, c0 = 8 * p, cnt = W - c0 < 8 ? W - c0 : 8;
        float f[8];
        for (int e = 0; e < cnt; ++e) f[e] = b2f(h[c0 + e]);
        for (int j = 0; j < R; ++j)
            for (int e = 0; e < cnt; ++e) acc[(size_t)t * (R + 1) + j] += fn[(int64_t)j * fp + c0 + e] * f[e];
        for (int e = 0; e < cnt; ++e) acc[(size_t)t * (R + 1) + R] += f[e] * f[e];
    }
    std::vector<float> s((size_t)R + 1, 0.0f);
    for (int k = 0; k <= R; ++k) {
        float tot = 0.0f;
        for (int w = 0; w < T / 32; ++w) {
            float l[32];
            for (int i = 0; i < 32; ++i) l[i] = acc[(size_t)(w * 32 + i) * (R + 1) + k];
            for (int d = 16; d >= 1; d >>= 1) {
                float nl[32];
                for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
                std::memcpy(l, nl, sizeof l);
            }
            tot += l[0];
        }
        s[(size_t)k] = tot;
    }
    const float inv = 1.0f / sqrtf(s[(size_t)R] / (float)W + eps);
    float gate[16];
    for (int j = 0; j < hc; ++j) gate[j] = host_sigmoid(s[(size_t)j] * inv * scale[0] + base[j]) + heps;
    if (post)
        for (int j = 0; j < hc; ++j)
            prow[j] = pmag * host_sigmoid(s[(size_t)(hc + j)] * inv * scale[1] + base[hc + j]) + heps;
    for (int i = 0; i < n; ++i) {
        float v = 0.0f;
        for (int j = 0; j < hc; ++j) v += gate[j] * b2f(h[j * n + i]);
        x[i] = f2b(v);
    }
}

static RadTensor t2(void* p, uint32_t dt, int64_t M, int64_t n, int64_t ld) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2; t.shape[0] = M; t.shape[1] = n; t.stride[0] = ld; t.stride[1] = 1;
    return t;
}
static RadTensor t1(void* p, uint32_t dt, int64_t n) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
template <class T>
static std::vector<T> aligned_like(size_t n) { return std::vector<T>(n + 64); }

static void ihc_pre_order() {
    struct G { int64_t hc, n, M, pad; bool post; };
    for (const G& g : { G{ 4, 3072, 2, 0, true }, G{ 4, 3072, 1, 3, true }, G{ 4, 512, 3, 0, true },
                        G{ 2, 128, 2, 0, true }, G{ 4, 512, 2, 0, false }, G{ 3, 37, 2, 1, true } }) {
        Rng rng((uint32_t)(g.hc * 7 + g.n + g.pad));
        const int64_t W = g.hc * g.n, rows = g.post ? 2 * g.hc : g.hc, hp = W + g.pad, xp = g.n + g.pad;
        std::vector<uint16_t> h((size_t)(g.M * hp)), xk((size_t)(g.M * xp), 0x7e7e), xh((size_t)(g.M * g.n));
        for (auto& v : h) v = f2b(rng.n() * 2);
        std::vector<float> fn((size_t)(rows * W)), base((size_t)rows), scale = { 3.0f, 2.1f };
        for (auto& v : fn) v = rng.n() * 0.05f;
        for (auto& v : base) v = rng.n() * 0.7f;
        std::vector<float> pk((size_t)(g.M * g.hc), 0.0f), ph((size_t)(g.M * g.hc), 0.0f);
        RadTensor t[6] = { t2(h.data(), RAD_BF16, g.M, W, hp), t2(fn.data(), RAD_F32, rows, W, W),
                           t1(scale.data(), RAD_F32, g.post ? 2 : 1), t1(base.data(), RAD_F32, rows),
                           t2(xk.data(), RAD_BF16, g.M, g.n, xp),
                           g.post ? t2(pk.data(), RAD_F32, g.M, g.hc, g.hc) : RadTensor{} };
        std::vector<RadParam> p = { pi("M", g.M), pi("n", g.n), pi("hc", g.hc), pi("rows", rows),
                                    pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
        if (g.post) p.push_back(pf("post_mag", 2.0));
        RadArgs a{}; a.t = t; a.n_t = 6; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
        ++g_cases;
        char w[120];
        std::snprintf(w, sizeof w, "ihc_pre order hc %lld n %lld M %lld%s%s, bit for bit", (long long)g.hc,
                      (long long)g.n, (long long)g.M, g.post ? "" : " head", g.pad ? " pitched (scalar loads)" : "");
        if (r4d_rad_ihc_pre(&a, nullptr) != RAD_OK) { std::printf("FAIL %s: launch\n", w); ++g_fails; continue; }
        long bad = 0;
        for (int64_t r = 0; r < g.M; ++r) {
            host_ihc_pre_row(h.data() + r * hp, fn.data(), W, scale.data(), base.data(), (int)g.hc, (int)g.n,
                             g.post, 1e-5f, 1e-6f, 2.0f, xh.data() + r * g.n, ph.data() + r * g.hc);
            for (int64_t i = 0; i < g.n; ++i) bad += xk[(size_t)(r * xp + i)] != xh[(size_t)(r * g.n + i)];
            if (g.post) bad += std::memcmp(pk.data() + r * g.hc, ph.data() + r * g.hc, (size_t)g.hc * 4) != 0;
        }
        if (bad) { std::printf("FAIL %s: %ld differ\n", w, bad); ++g_fails; }
        else std::printf("ok   %s\n", w);
    }
}

/* ---- rmsnorm_ihc_junction against its chain, byte for byte: the add (host, add_bf16's
 * arithmetic), rmsnorm in rmsnorm_bf16's order (host: kR4dSumLanes' partition, the wave butterfly,
 * the waves in order, rsqrtf as the emulator has it), the emulated ihc_post and ihc_pre kernels,
 * rmsnorm again. Every form: with and without the add, `y` / `x` stored or not, tail 0, the head. */
static void host_rmsnorm_row(const uint16_t* x, const float* g, int n, float eps, uint16_t* y) {
    float part[256];
    for (int t = 0; t < 256; ++t) {
        float ss = 0.0f;
        for (int i = t; i < n; i += 256) { const float v = b2f(x[i]); ss += v * v; }
        part[t] = ss;
    }
    float total = 0.0f;
    for (int w = 0; w < 8; ++w) {
        float l[32];
        for (int i = 0; i < 32; ++i) l[i] = part[w * 32 + i];
        for (int d = 16; d >= 1; d >>= 1) {
            float nl[32];
            for (int i = 0; i < 32; ++i) nl[i] = l[i] + l[i ^ d];
            std::memcpy(l, nl, sizeof l);
        }
        total += l[0];
    }
    const float s = 1.0f / std::sqrt(total / (float)n + eps);
    const float wadd = 0.0f, mult = 1.0f;
    for (int i = 0; i < n; ++i) y[i] = f2b(b2f(x[i]) * s * (g[i] + wadd) * mult);
}

static void ihc_junction() {
    struct G { int64_t hc, n, M; bool add, ys, xs, tail, head; };
    const G cases[] = {
        { 4, 3072, 1, false, false, false, true, false }, { 4, 3072, 6, true, true, true, true, false },
        { 4, 512, 17, true, false, true, true, false },   { 4, 512, 6, false, true, false, true, true },
        { 2, 128, 6, true, true, true, false, false },    { 2, 128, 64, false, false, false, true, false },
        { 4, 512, 3, true, false, false, true, true },    { 8, 64, 2, false, true, true, true, false },
        { 2, 128, 1024, true, true, false, true, false },
    };
    for (const G& g : cases) {
        Rng rng((uint32_t)(g.hc * 13 + g.n + g.M));
        const int64_t M = g.M, n = g.n, hc = g.hc, W = hc * n, rows = g.head ? hc : 2 * hc;
        const bool post = !g.head;
        std::vector<uint16_t> A((size_t)(M * n)), B((size_t)(M * n)), H((size_t)(M * W));
        for (auto& v : A) v = f2b(rng.n() * 1.5f);
        for (auto& v : B) v = f2b(rng.n() * 1.5f);
        for (auto& v : H) v = f2b(rng.n() * 2);
        std::vector<float> wp((size_t)n), wn((size_t)n), P((size_t)(M * hc)), fn((size_t)(rows * W)),
            base((size_t)rows), scale = { 3.0f, 2.1f };
        for (auto& v : wp) v = 1.0f + rng.n() * 0.4f;
        for (auto& v : wn) v = 1.0f + rng.n() * 0.4f;
        for (auto& v : P) v = 0.5f + rng.u();
        for (auto& v : fn) v = rng.n() * 0.05f;
        for (auto& v : base) v = rng.n() * 0.7f;
        /* the fused kernel */
        auto Ak = A, Hk = H;
        auto Pk = P;
        std::vector<uint16_t> XNk((size_t)(M * n), 0x7e7e), Yk((size_t)(M * n), 0x7e7e), Xk((size_t)(M * n), 0x7e7e);
        RadTensor t[12] = {
            t2(Ak.data(), RAD_BF16, M, n, n), g.add ? t2(B.data(), RAD_BF16, M, n, n) : RadTensor{},
            t1(wp.data(), RAD_F32, n), t2(Pk.data(), RAD_F32, M, hc, hc), t2(Hk.data(), RAD_BF16, M, W, W),
            g.tail ? t2(fn.data(), RAD_F32, rows, W, W) : RadTensor{},
            g.tail ? t1(scale.data(), RAD_F32, post ? 2 : 1) : RadTensor{},
            g.tail ? t1(base.data(), RAD_F32, rows) : RadTensor{},
            g.tail ? t1(wn.data(), RAD_F32, n) : RadTensor{},
            g.tail ? t2(XNk.data(), RAD_BF16, M, n, n) : RadTensor{},
            g.ys ? t2(Yk.data(), RAD_BF16, M, n, n) : RadTensor{},
            g.xs && g.tail ? t2(Xk.data(), RAD_BF16, M, n, n) : RadTensor{} };
        std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("hc", hc), pi("rows", rows), pf("eps", 1e-6),
                                    pf("ihc_eps", 1e-5), pf("hc_eps", 1e-6), pi("tail", g.tail ? 1 : 0),
                                    ps("dtype", "bf16") };
        if (post) p.push_back(pf("post_mag", 2.0));
        RadArgs a{}; a.t = t; a.n_t = 12; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
        char w[160];
        std::snprintf(w, sizeof w, "rmsnorm_ihc_junction hc %lld n %lld M %lld%s%s%s%s vs the chain, bit for bit",
                      (long long)hc, (long long)n, (long long)M, g.add ? " add" : "", g.tail ? "" : " tail 0",
                      g.head ? " head" : "", g.ys || g.xs ? " y/x stored" : "");
        ++g_cases;
        if (r4d_rad_rmsnorm_ihc_junction(&a, nullptr) != RAD_OK) { std::printf("FAIL %s: launch\n", w); ++g_fails; continue; }
        /* the chain */
        auto Ac = A, Hc = H;
        auto Pc = P;
        std::vector<uint16_t> Yc((size_t)(M * n)), Xc((size_t)(M * n)), XNc((size_t)(M * n));
        if (g.add)
            for (size_t i = 0; i < Ac.size(); ++i) Ac[i] = f2b(b2f(Ac[i]) + b2f(B[i]));
        for (int64_t r = 0; r < M; ++r) host_rmsnorm_row(Ac.data() + r * n, wp.data(), (int)n, 1e-6f, Yc.data() + r * n);
        {
            RadTensor tp[3] = { t2(Yc.data(), RAD_BF16, M, n, n), t2(Pc.data(), RAD_F32, M, hc, hc),
                                t2(Hc.data(), RAD_BF16, M, W, W) };
            std::vector<RadParam> q = { pi("M", M), pi("n", n), pi("hc", hc), ps("dtype", "bf16") };
            RadArgs b{}; b.t = tp; b.n_t = 3; b.p = q.data(); b.n_p = (int)q.size(); b.world_size = 1;
            if (r4d_rad_ihc_post(&b, nullptr) != RAD_OK) { std::printf("FAIL %s: chain ihc_post\n", w); ++g_fails; continue; }
        }
        if (g.tail) {
            RadTensor tp[6] = { t2(Hc.data(), RAD_BF16, M, W, W), t2(fn.data(), RAD_F32, rows, W, W),
                                t1(scale.data(), RAD_F32, post ? 2 : 1), t1(base.data(), RAD_F32, rows),
                                t2(Xc.data(), RAD_BF16, M, n, n),
                                post ? t2(Pc.data(), RAD_F32, M, hc, hc) : RadTensor{} };
            std::vector<RadParam> q = { pi("M", M), pi("n", n), pi("hc", hc), pi("rows", rows),
                                        pf("eps", 1e-5), pf("hc_eps", 1e-6), ps("dtype", "bf16") };
            if (post) q.push_back(pf("post_mag", 2.0));
            RadArgs b{}; b.t = tp; b.n_t = 6; b.p = q.data(); b.n_p = (int)q.size(); b.world_size = 1;
            if (r4d_rad_ihc_pre(&b, nullptr) != RAD_OK) { std::printf("FAIL %s: chain ihc_pre\n", w); ++g_fails; continue; }
            for (int64_t r = 0; r < M; ++r) host_rmsnorm_row(Xc.data() + r * n, wn.data(), (int)n, 1e-6f, XNc.data() + r * n);
        }
        long bad = 0;
        auto cmpv = [&](const void* x, const void* y, size_t bytes, const char* what) {
            if (std::memcmp(x, y, bytes)) { if (!bad) std::printf("  first difference in %s\n", what); ++bad; }
        };
        cmpv(Ak.data(), Ac.data(), Ac.size() * 2, "a");
        cmpv(Hk.data(), Hc.data(), Hc.size() * 2, "h");
        cmpv(Pk.data(), Pc.data(), Pc.size() * 4, "post");
        if (g.tail) cmpv(XNk.data(), XNc.data(), XNc.size() * 2, "xn");
        if (g.ys) cmpv(Yk.data(), Yc.data(), Yc.size() * 2, "y");
        else for (auto v : Yk) if (v != 0x7e7e) { cmpv("a", "b", 1, "y written though absent"); break; }
        if (g.xs && g.tail) cmpv(Xk.data(), Xc.data(), Xc.size() * 2, "x");
        if (bad) { std::printf("FAIL %s\n", w); ++g_fails; }
        else std::printf("ok   %s\n", w);
    }
}

/* ---- K-46: the junction's rows in groups (r4d_ihc_bf16.hip's ihc_junction_group_kernel over
 * r4d_ihc.h's junction_walk, long prefill only) against upstream's one-row kernel, byte for byte.
 * The reference is the launch at nblocks -1 (ihc_junction_kernel, one row a block, junction_row);
 * each other count runs the grouped kernel over that many blocks where it exists (tail, hc 2/4/8,
 * G rows in 48 KiB) and the one-row kernel elsewhere: 0 (the op's default: grouped from 2048 rows),
 * M (one row a block inside the grouped kernel), ceil(M / 2) (one group a block), ceil(M / 4),
 * the two-rank twin's min(M, 512) and the four-rank twin's min(M, 64) partitions (and each one
 * fewer, the off-cliff step), and 1 (every row in one block: groups, then a lone tail row). Rows 1 to
 * 2049, odd counts among them; hc 2, 3, 4, 5, 7 and 8, the head, the add, tail 0 (no group), the
 * widest grouped row (n 12288) and one past it (16384), y/x stored and G-3b's codes. The bytes
 * compared: a, h, post, xn, y, x, xq and xs. */
extern "C" int r4d_rmsnorm_ihc_junction_nb(const RadArgs* a, RadStream s, long nblocks);
static void ihc_junction_groups() {
    struct G { int64_t hc, n, M; bool add, ys, tail, head, codes; };
    const G cases[] = {
        { 4, 3072, 1, true, true, true, false, true },     { 4, 3072, 7, true, true, true, false, true },
        { 4, 1024, 13, false, false, true, false, false }, { 4, 512, 64, true, true, true, true, false },
        { 2, 128, 130, false, true, true, false, true },   { 8, 256, 9, true, true, true, false, false },
        { 8, 128, 5, false, false, true, true, true },     { 2, 128, 6, true, true, false, false, false },
        { 4, 8192, 7, false, true, true, false, false },   { 4, 12288, 5, true, false, true, false, true },
        { 4, 16384, 3, true, false, true, false, false },  { 2, 128, 1026, true, false, true, false, true },
        { 4, 256, 2048, false, true, true, false, false }, { 4, 128, 2049, true, true, true, false, true },
        { 8, 128, 2049, false, true, true, false, false }, { 2, 128, 2048, true, false, true, true, false },
        { 4, 128, 2048, true, true, false, false, false }, /* tail 0 at the grouped height: one-row */
        { 7, 64, 6, true, true, true, false, false },      { 3, 64, 2048, true, true, true, false, false },
        { 5, 64, 11, false, true, true, true, false },     /* odd hc: one row a block at every count */
    };
    for (const G& g : cases) {
        Rng rng((uint32_t)(g.hc * 31 + g.n + 7 * g.M));
        const int64_t M = g.M, n = g.n, hc = g.hc, W = hc * n, rows = g.head ? hc : 2 * hc;
        const bool post = !g.head, codes = g.codes && g.tail;
        std::vector<uint16_t> A((size_t)(M * n)), B((size_t)(M * n)), H((size_t)(M * W));
        for (auto& v : A) v = f2b(rng.n() * 1.5f);
        for (auto& v : B) v = f2b(rng.n() * 1.5f);
        for (auto& v : H) v = f2b(rng.n() * 2);
        std::vector<float> wp((size_t)n), wn((size_t)n), P((size_t)(M * hc)), fn((size_t)(rows * W)),
            base((size_t)rows), scale = { 3.0f, 2.1f };
        for (auto& v : wp) v = 1.0f + rng.n() * 0.4f;
        for (auto& v : wn) v = 1.0f + rng.n() * 0.4f;
        for (auto& v : P) v = 0.5f + rng.u();
        for (auto& v : fn) v = rng.n() * 0.05f;
        for (auto& v : base) v = rng.n() * 0.7f;
        struct Out { std::vector<uint16_t> a, h, xn, y, x; std::vector<float> p, xs; std::vector<int8_t> xq; };
        auto run = [&](long nb, Out& o) {
            o.a = A; o.h = H; o.p = P;
            o.xn.assign((size_t)(M * n), 0x7e7e); o.y.assign((size_t)(M * n), 0x7e7e);
            o.x.assign((size_t)(M * n), 0x7e7e);
            o.xq.assign((size_t)(M * n), 0x5a); o.xs.assign((size_t)(M * (n / 128 > 0 ? n / 128 : 1)), -7.0f);
            RadTensor t[14] = {
                t2(o.a.data(), RAD_BF16, M, n, n), g.add ? t2(B.data(), RAD_BF16, M, n, n) : RadTensor{},
                t1(wp.data(), RAD_F32, n), t2(o.p.data(), RAD_F32, M, hc, hc), t2(o.h.data(), RAD_BF16, M, W, W),
                g.tail ? t2(fn.data(), RAD_F32, rows, W, W) : RadTensor{},
                g.tail ? t1(scale.data(), RAD_F32, post ? 2 : 1) : RadTensor{},
                g.tail ? t1(base.data(), RAD_F32, rows) : RadTensor{},
                g.tail ? t1(wn.data(), RAD_F32, n) : RadTensor{},
                g.tail ? t2(o.xn.data(), RAD_BF16, M, n, n) : RadTensor{},
                g.ys ? t2(o.y.data(), RAD_BF16, M, n, n) : RadTensor{},
                g.ys && g.tail ? t2(o.x.data(), RAD_BF16, M, n, n) : RadTensor{},
                codes ? t2(o.xq.data(), RAD_I8, M, n, n) : RadTensor{},
                codes ? t2(o.xs.data(), RAD_F32, M, n / 128, n / 128) : RadTensor{} };
            std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("hc", hc), pi("rows", rows), pf("eps", 1e-6),
                                        pf("ihc_eps", 1e-5), pf("hc_eps", 1e-6), pi("tail", g.tail ? 1 : 0),
                                        ps("dtype", "bf16") };
            if (post) p.push_back(pf("post_mag", 2.0));
            RadArgs a{}; a.t = t; a.n_t = 14; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
            return r4d_rmsnorm_ihc_junction_nb(&a, nullptr, nb);
        };
        Out ref;
        char w[200];
        std::snprintf(w, sizeof w, "junction rows in groups hc %lld n %lld M %lld%s%s%s%s%s", (long long)hc,
                      (long long)n, (long long)M, g.add ? " add" : "", g.tail ? "" : " tail 0", g.head ? " head" : "",
                      g.ys ? " y/x stored" : "", codes ? " codes" : "");
        ++g_cases;
        if (run(-1, ref) != RAD_OK) { std::printf("FAIL %s: the one-row launch\n", w); ++g_fails; continue; }
        std::vector<long> nbs = { 0, (long)M, (long)((M + 1) / 2), (long)((M + 3) / 4), std::min<long>(M, 512),
                                  std::min<long>(M, 511), std::min<long>(M, 64), std::min<long>(M, 63), 1 };
        std::sort(nbs.begin(), nbs.end());
        nbs.erase(std::unique(nbs.begin(), nbs.end()), nbs.end());
        long bad = 0;
        std::string counts;
        for (long nb : nbs) {
            Out o;
            if (run(nb, o) != RAD_OK) { std::printf("  launch at %ld block(s) refused\n", nb); ++bad; continue; }
            auto cmpv = [&](const void* x, const void* y, size_t bytes, const char* what) {
                if (std::memcmp(x, y, bytes)) { std::printf("  %ld block(s): %s differs\n", nb, what); ++bad; }
            };
            cmpv(o.a.data(), ref.a.data(), ref.a.size() * 2, "a");
            cmpv(o.h.data(), ref.h.data(), ref.h.size() * 2, "h");
            cmpv(o.p.data(), ref.p.data(), ref.p.size() * 4, "post");
            cmpv(o.xn.data(), ref.xn.data(), ref.xn.size() * 2, "xn");
            cmpv(o.y.data(), ref.y.data(), ref.y.size() * 2, "y");
            cmpv(o.x.data(), ref.x.data(), ref.x.size() * 2, "x");
            cmpv(o.xq.data(), ref.xq.data(), ref.xq.size(), "xq");
            cmpv(o.xs.data(), ref.xs.data(), ref.xs.size() * 4, "xs");
            counts += nb == 0 ? std::string(" default") : " " + std::to_string(nb);
        }
        /* the reference wrote what it should: h moved, and xn (tail) */
        if (!std::memcmp(ref.h.data(), H.data(), H.size() * 2)) { std::printf("  h unwritten\n"); ++bad; }
        if (g.tail) for (auto v : ref.xn) if (v == 0x7e7e) { std::printf("  xn unwritten\n"); ++bad; break; }
        if (bad) { std::printf("FAIL %s\n", w); ++g_fails; }
        else std::printf("ok   %s vs the one-row kernel, at%s block(s)\n", w, counts.empty() ? " no other" : counts.c_str());
    }
}

int main() {
    ihc_pre();
    ihc_post();
    ihc_pre_order();
    ihc_junction();
    ihc_junction_groups();
    gate_mul();
    mla_kv();
    engram_ids();
    engram_norm_gate();
    dfly_sel();
    std::printf("%d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
