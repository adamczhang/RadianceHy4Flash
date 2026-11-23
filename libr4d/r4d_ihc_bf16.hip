// r4d_ihc_bf16.hip -- Hy4-flash's independent hyper-connections and the sigmoid-gated product.
//
//   ihc_pre   f = f32(h); r = 1/sqrt(sum f^2 / (hc n) + eps); c[j] = (fn[j] . f) r
//             pre[j]  = sigmoid(c[j] scale[0] + base[j]) + hc_eps               j < hc
//             x       = bf16(sum_j pre[j] f[j])
//             post[j] = post_mag sigmoid(c[hc+j] scale[1] + base[hc+j]) + hc_eps  (when `post` is
//                       present: rows == 2 hc and scale has two entries)
//   ihc_post  h[j][i] = bf16(post[j] f32(y[i]) + f32(h[j][i])), in place on the hc streams
//   gate_mul  y = bf16(f32(a) sigmoid(f32(g))), one rounding
//
// libref/ref_hy4.cpp is the oracle, kernels/ihc.md the design. Written for this repository; the
// semantics are HYV4FlashHyperConnection / HyperHead as the design note states them.
//
// ihc_pre: one workgroup of 512 threads a token row, the order r4d_ihc.h states (16-byte pieces of
// the flattened row, rows + 1 f32 sums a thread, r4d_xor_sum over the lanes, the 16 waves in order).
// It replaced (WP16.2) a kernel whose 256 threads strided single columns with 2-byte loads, 18.7 us
// a row at n = 3072; that changed the sums' order once, inside the bound below. One thread forms r,
// the gates and `post`; every thread then forms x for its 8-column groups. libref sums in ascending
// order (the sum of squares in double), so the kernel agrees with it to rounding and not in bytes:
// relative 1e-5 on c, x within one bf16 ulp except where a sum straddles a rounding boundary, post
// within 2e-4. The order does not depend on M, the pitch or alignment (the scalar-load form for odd
// widths or pointers runs it too), so a row's bytes do not depend on the batch. The fused junction
// (below) and ar_rmsnorm_ihc_junction (r4d_ar_ihc_junction.hip) run the same order.
// LDS: (16 * (rows + 1) + hc) floats, under 1.2 KiB. No scratch.
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_chain.h"
#include "r4d_ihc.h"
#include "r4d_launch.h"
#include <climits>
#include <cmath>
#include <cstdint>

namespace {

using namespace r4d_ihc;

__device__ __forceinline__ float ihc_sigmoid(float x) { return r4d_ihc::sigmoid(x); }

// VEC: n % 8 == 0, h, x and fn rows 16-byte aligned with whole-word pitches (16-byte loads of h
// and x, float4 loads of fn). Otherwise the same pieces through scalar loads.
template <int HC, bool POST, bool VEC>
__device__ __forceinline__ void ihc_pre_body(
    const uint16_t* __restrict__ h, const float* __restrict__ fn, const float* __restrict__ scale,
    const float* __restrict__ base, uint16_t* __restrict__ x, float* __restrict__ post,
    uint32_t n, float eps, float heps, float pmag, uint64_t hp, uint64_t fp, uint64_t xp,
    uint64_t pp) {
    constexpr int R = POST ? 2 * HC : HC;
    __shared__ float part[kWaves][R + 1];
    __shared__ float gate[HC];
    const uint32_t t = threadIdx.x;
    const uint64_t row = blockIdx.x;
    const uint32_t width = (uint32_t)HC * n;
    const uint16_t* __restrict__ hr = h + row * hp;

    float acc[R + 1];
#pragma unroll
    for (int k = 0; k <= R; ++k) acc[k] = 0.0f;
    const uint32_t pieces = (width + 7u) / 8u;
    for (uint32_t p = t; p < pieces; p += kThreads) {
        const uint32_t c0 = 8u * p;
        float f[8];
        if (VEC) {
            unpack8(*(const uint4*)(hr + c0), f);
            acc_piece<R, true>(acc, f, 8u, fn, fp, c0);
        } else {
            const uint32_t cnt = width - c0 < 8u ? width - c0 : 8u;
            for (uint32_t e = 0; e < cnt; ++e) f[e] = bf16_to_f32(hr[c0 + e]);
            acc_piece<R, false>(acc, f, cnt, fn, fp, c0);
        }
    }
    finish<HC, POST>(acc, part, gate, scale, base, POST ? post + row * pp : nullptr, width, eps,
                     heps, pmag);
    uint16_t* __restrict__ xr = x + row * xp;
    if (VEC) {
        collapse8<HC>(hr, gate, n, [&](uint32_t q, uint4 w) { *(uint4*)(xr + 8u * q) = w; });
    } else {
        for (uint32_t i = t; i < n; i += kThreads) {
            float v = 0.0f;
#pragma unroll
            for (int j = 0; j < HC; ++j) v += gate[j] * bf16_to_f32(hr[(uint32_t)j * n + i]);
            xr[i] = f32_to_bf16(v);
        }
    }
}

#define R4D_IHC_PRE_PARAMS                                                                         \
    const uint16_t* __restrict__ h, const float* __restrict__ fn, const float* __restrict__ scale, \
    const float* __restrict__ base, uint16_t* __restrict__ x, float* __restrict__ post,            \
    uint32_t n, float eps, float heps, float pmag, uint64_t hp, uint64_t fp, uint64_t xp,          \
    uint64_t pp, uint32_t M
// CH `r4dchain`: the chained form (r4d_chain.h); every block reaches the done. Scale and base are
// read through scalar loads, so the wait is the full one. Rows at or past M are the ones the launch
// adds to step off the full-card cliff (r4d_cliff_rows: 16 waves a row lands every 128th M on it).
template <int HC, bool POST, bool VEC, class CH>
__global__ __launch_bounds__(kThreads) void ihc_pre_kernel(R4D_IHC_PRE_PARAMS) {
    if (CH::on) r4d_chain_wait();
    if (blockIdx.x < M)
        ihc_pre_body<HC, POST, VEC>(h, fn, scale, base, x, post, n, eps, heps, pmag, hp, fp, xp, pp);
    if (CH::on) r4d_chain_done();
}

// rmsnorm_ihc_junction: `[add] -> rmsnorm -> ihc_post -> [ihc_pre -> rmsnorm]` a token row a
// workgroup of 512 threads (r4d_ihc.h's junction_row), byte-identical to add_bf16, rmsnorm_bf16,
// ihc_post_bf16, ihc_pre_bf16 and rmsnorm_bf16 in sequence. ADD: the row is bf16(a + b), stored
// back into `a` as add_bf16 stores it. TAIL 0 stops behind the write (an Engram in front of the
// next layer writes h between). LDS: n bf16 (the row, then x) + about 0.9 KiB; no scratch.
template <int HC, bool POST, bool TAIL, bool ADD>
__device__ __forceinline__ void ihc_junction_body(
    uint16_t* __restrict__ a, const uint16_t* __restrict__ b, JuncRow r, uint64_t ap, uint64_t bp,
    uint64_t pp, uint64_t hp, uint64_t xnp, uint64_t yp, uint64_t xp) {
    R4D_DYN_LDS(uint4, jbuf);
    __shared__ JuncLds<HC, POST> L;
    uint16_t* const ybuf = (uint16_t*)jbuf;
    const uint64_t row = blockIdx.x;
    uint16_t* __restrict__ ar = a + row * ap;
    for (uint32_t q = threadIdx.x; q < r.n / 8u; q += kThreads) {
        uint4 w = *(const uint4*)(ar + 8u * q);
        if (ADD) {
            float u[8], v[8];
            unpack8(w, u);
            unpack8(*(const uint4*)(b + row * bp + 8u * q), v);
            uint16_t o[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = f32_to_bf16(u[e] + v[e]);
            w = pack8(o);
            *(uint4*)(ar + 8u * q) = w;
        }
        jbuf[q] = w;
    }
    r.post_row += row * pp;
    r.hr += row * hp;
    if (TAIL) r.xnr += row * xnp;
    if (r.cq) { r.cq += row * r.cqp; r.cs += row * r.csp; }
    if (r.yr) r.yr += row * yp;
    if (r.xr) r.xr += row * xp;
    junction_row<HC, POST, TAIL>(L, ybuf, r);
}

template <int HC, bool POST, bool TAIL, bool ADD, class CH>
__global__ __launch_bounds__(kThreads) void ihc_junction_kernel(
    uint16_t* __restrict__ a, const uint16_t* __restrict__ b, JuncRow r, uint64_t ap, uint64_t bp,
    uint64_t pp, uint64_t hp, uint64_t xnp, uint64_t yp, uint64_t xp, uint32_t M) {
    if (CH::on) r4d_chain_wait();
    if (blockIdx.x < M) ihc_junction_body<HC, POST, TAIL, ADD>(a, b, r, ap, bp, pp, hp, xnp, yp, xp);
    if (CH::on) r4d_chain_done();
}
// The grouped instance (K-46): long prefill only (r4d_ihc.h's junction_grouped), the tail always.
// Block bk of `nbw` working blocks owns the rows [bk * M / nbw, (bk + 1) * M / nbw) and takes them G
// at a time (junction_walk); the op's launch gives ceil(M / G) blocks, one group each. A row's bytes
// are ihc_junction_kernel's whatever nbw. The register allocation is bounded to JW waves a SIMD
// (junction_group_waves) so the group's sums cannot take the occupancy the one-row kernel has.
// LDS: G rows of n bf16 + G JuncLds.
template <int HC, bool POST, bool ADD, class CH, int G>
__global__ __launch_bounds__(kThreads)
__attribute__((amdgpu_waves_per_eu((junction_group_waves(HC, POST)), 16))) void ihc_junction_group_kernel(
    uint16_t* __restrict__ a, const uint16_t* __restrict__ b, JuncRow r, uint64_t ap, uint64_t bp,
    uint64_t pp, uint64_t hp, uint64_t xnp, uint64_t yp, uint64_t xp, uint32_t M, uint32_t nbw) {
    if (CH::on) r4d_chain_wait();
    if (blockIdx.x < nbw) {
        R4D_DYN_LDS(uint4, jbuf);
        __shared__ JuncLds<HC, POST> L[G];
        const uint64_t bk = blockIdx.x;
        const int r0 = (int)(bk * M / nbw), r1 = (int)((bk + 1) * M / nbw);
        junction_walk<HC, POST, G>(L, (uint16_t*)jbuf, r, r0, r1, pp, hp, xnp, yp, xp,
                                   [&](int row, uint16_t* buf) {
            uint16_t* __restrict__ ar = a + (uint64_t)row * ap;
            for (uint32_t q = threadIdx.x; q < r.n / 8u; q += kThreads) {
                uint4 w = *(const uint4*)(ar + 8u * q);
                if (ADD) {
                    float u[8], v[8];
                    unpack8(w, u);
                    unpack8(*(const uint4*)(b + (uint64_t)row * bp + 8u * q), v);
                    uint16_t o[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) o[e] = f32_to_bf16(u[e] + v[e]);
                    w = pack8(o);
                    *(uint4*)(ar + 8u * q) = w;
                }
                ((uint4*)buf)[q] = w;
            }
        });
    }
    if (CH::on) r4d_chain_done();
}
// h[j][i] = bf16(post[j] * y[i] + h[j][i]). The product is its own f32 rounding, as libref's.
// Eight columns of every stream a thread when rows and pointers allow 16-byte accesses.
template <bool VEC>
__device__ __forceinline__ void ihc_post_body(
    const uint16_t* __restrict__ y, const float* __restrict__ post, uint16_t* __restrict__ h,
    uint32_t n, uint32_t hc, uint64_t yp, uint64_t pp, uint64_t hp) {
    const uint64_t row = blockIdx.y;
    const uint16_t* __restrict__ yr = y + row * yp;
    const float* __restrict__ pr = post + row * pp;
    uint16_t* __restrict__ hr = h + row * hp;
    if (VEC) {
        const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
        if (i >= n) return;
        const uint4 yv = *(const uint4*)(yr + i);
        float yf[8];
        const uint32_t yw[4] = { yv.x, yv.y, yv.z, yv.w };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            yf[2 * k]     = __uint_as_float(yw[k] << 16);
            yf[2 * k + 1] = __uint_as_float(yw[k] & 0xffff0000u);
        }
        for (uint32_t j = 0; j < hc; ++j) {
            const float p = pr[j];
            uint4 hv = *(const uint4*)(hr + (uint64_t)j * n + i);
            uint32_t hw[4] = { hv.x, hv.y, hv.z, hv.w };
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float lo = __uint_as_float(hw[k] << 16), hi = __uint_as_float(hw[k] & 0xffff0000u);
                const float a = __fmul_rn(p, yf[2 * k]), b = __fmul_rn(p, yf[2 * k + 1]);
                hw[k] = (uint32_t)f32_to_bf16(__fadd_rn(a, lo)) |
                        ((uint32_t)f32_to_bf16(__fadd_rn(b, hi)) << 16);
            }
            *(uint4*)(hr + (uint64_t)j * n + i) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
        }
    } else {
        const uint32_t i = blockIdx.x * 256u + threadIdx.x;
        if (i >= n) return;
        const float yf = bf16_to_f32(yr[i]);
        for (uint32_t j = 0; j < hc; ++j) {
            uint16_t* hp_ = hr + (uint64_t)j * n + i;
            *hp_ = f32_to_bf16(__fadd_rn(__fmul_rn(pr[j], yf), bf16_to_f32(*hp_)));
        }
    }
}

// CH `r4dchain`: the chained form (r4d_chain.h); the body keeps its early returns, the wrapper
// brackets it so every block reaches the done.
template <bool VEC, class CH>
__global__ __launch_bounds__(256) void ihc_post_kernel(
    const uint16_t* __restrict__ y, const float* __restrict__ post, uint16_t* __restrict__ h,
    uint32_t n, uint32_t hc, uint64_t yp, uint64_t pp, uint64_t hp) {
    if (CH::on) r4d_chain_wait();
    ihc_post_body<VEC>(y, post, h, n, hc, yp, pp, hp);
    if (CH::on) r4d_chain_done();
}

// y = bf16(a * sigmoid(g)).
template <bool VEC>
__device__ __forceinline__ void gate_mul_body(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ g, uint16_t* __restrict__ y,
    uint32_t n, uint64_t ap, uint64_t gp, uint64_t yp) {
    const uint64_t row = blockIdx.y;
    if (VEC) {
        const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 8u;
        if (i >= n) return;
        const uint4 av = *(const uint4*)(a + row * ap + i), gv = *(const uint4*)(g + row * gp + i);
        const uint32_t aw[4] = { av.x, av.y, av.z, av.w }, gw[4] = { gv.x, gv.y, gv.z, gv.w };
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a0 = __uint_as_float(aw[k] << 16), a1 = __uint_as_float(aw[k] & 0xffff0000u);
            const float g0 = __uint_as_float(gw[k] << 16), g1 = __uint_as_float(gw[k] & 0xffff0000u);
            o[k] = (uint32_t)f32_to_bf16(a0 * ihc_sigmoid(g0)) |
                   ((uint32_t)f32_to_bf16(a1 * ihc_sigmoid(g1)) << 16);
        }
        *(uint4*)(y + row * yp + i) = make_uint4(o[0], o[1], o[2], o[3]);
    } else {
        const uint32_t i = blockIdx.x * 256u + threadIdx.x;
        if (i >= n) return;
        y[row * yp + i] = f32_to_bf16(bf16_to_f32(a[row * ap + i]) *
                                      ihc_sigmoid(bf16_to_f32(g[row * gp + i])));
    }
}

template <bool VEC, class CH>
__global__ __launch_bounds__(256) void gate_mul_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ g, uint16_t* __restrict__ y,
    uint32_t n, uint64_t ap, uint64_t gp, uint64_t yp) {
    if (CH::on) r4d_chain_wait();
    gate_mul_body<VEC>(a, g, y, n, ap, gp, yp);
    if (CH::on) r4d_chain_done();
}

bool ihc_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
bool ihc_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
bool ihc_vec(const void* p, uint64_t pitch) { return ((uintptr_t)p & 15) == 0 && pitch % 8 == 0; }

}  // namespace

enum { IHC_H = 0, IHC_FN, IHC_SCALE, IHC_BASE, IHC_X, IHC_POST };

extern "C" int r4d_rad_ihc_pre(const RadArgs* a, RadStream s) {
    const RadTensor* h = r4d_opt(a, IHC_H);
    const RadTensor* fn = r4d_opt(a, IHC_FN);
    const RadTensor* sc = r4d_opt(a, IHC_SCALE);
    const RadTensor* bs = r4d_opt(a, IHC_BASE);
    const RadTensor* x = r4d_opt(a, IHC_X);
    const RadTensor* post = r4d_opt(a, IHC_POST);
    if (!h || !fn || !sc || !bs || !x) return R4D_NO(RAD_E_INVAL);
    if (!ihc_bf16(h) || !ihc_bf16(x) || !ihc_f32(fn) || !ihc_f32(sc) || !ihc_f32(bs) ||
        (post && !ihc_f32(post)))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0),
                    rows = r4d_geti_or(a, "rows", 0);
    const float eps = (float)r4d_getf(a, "eps", 0), heps = (float)r4d_getf(a, "hc_eps", 0);
    if (n < 1 || n > 16384 || hc < 2 || hc > 8 || !std::isfinite(eps) || eps <= 0 ||
        !std::isfinite(heps) || heps < 0)
        return R4D_NO(RAD_E_SHAPE);
    const bool has_post = post != nullptr;
    if (rows != (has_post ? 2 * hc : hc)) return R4D_NO(RAD_E_SHAPE);
    float pmag = 0.0f;
    if (has_post) {
        if (r4d_getf(a, "post_mag", -1e30) == -1e30) return R4D_NO(RAD_E_INVAL);
        pmag = (float)r4d_getf(a, "post_mag", 0);
        if (!std::isfinite(pmag)) return R4D_NO(RAD_E_SHAPE);
    }
    const long long W = hc * n;
    long long M = 0, Mx = 0, Mp = 0;
    const long long hp = r4d_rows_pitch(h, W, &M), xp = r4d_rows_pitch(x, n, &Mx);
    if (hp < 0 || xp < 0) return R4D_NO(RAD_E_STRIDE);
    long long pp = 0;
    if (has_post) {
        pp = r4d_rows_pitch(post, hc, &Mp);
        if (pp < 0) return R4D_NO(RAD_E_STRIDE);
    }
    if (M < 1 || M > 65535 || Mx != M || (has_post && Mp != M)) return R4D_NO(RAD_E_SHAPE);
    if (fn->rank != 2 || fn->shape[0] != rows || fn->shape[1] != W || fn->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(sc) != (has_post ? 2 : 1) || r4d_numel(bs) != rows) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(sc) || !r4d_contig(bs)) return R4D_NO(RAD_E_STRIDE);

    const bool vec = n % 8 == 0 && ihc_vec(h->data, hp) && ihc_vec(x->data, xp) &&
                     ((uintptr_t)fn->data & 15) == 0 && fn->stride[0] % 4 == 0;
    /* The chained form at the vector form only (the real geometry); the scalar one stays plain. */
    const bool ch = vec && r4d_geti_or(a, "chain", 0) != 0;
    const uint32_t grid = r4d_cliff_rows((uint32_t)M, 1, kWaves);
#define R4D_IHC_K(HC_, POST_, VEC_, CH_)                                                           \
    R4D_EMIT(s, (ihc_pre_kernel<HC_, POST_, VEC_, CH_>), dim3(grid), dim3(kThreads), 0,            \
             (const uint16_t*)h->data, (const float*)fn->data, (const float*)sc->data,            \
             (const float*)bs->data, (uint16_t*)x->data, (float*)(has_post ? post->data : nullptr), \
             (uint32_t)n, eps, heps, pmag, (uint64_t)hp, (uint64_t)fn->stride[0], (uint64_t)xp,   \
             (uint64_t)pp, (uint32_t)M)
#define R4D_IHC(HC_, POST_)                                                                        \
    (ch ? R4D_IHC_K(HC_, POST_, true, r4dchain)                                                    \
        : vec ? R4D_IHC_K(HC_, POST_, true, R4dUnchained) : R4D_IHC_K(HC_, POST_, false, R4dUnchained))
#define R4D_IHC_HC(HC_) return has_post ? R4D_IHC(HC_, true) : R4D_IHC(HC_, false)
    switch (hc) {
        case 2: R4D_IHC_HC(2);
        case 3: R4D_IHC_HC(3);
        case 4: R4D_IHC_HC(4);
        case 5: R4D_IHC_HC(5);
        case 6: R4D_IHC_HC(6);
        case 7: R4D_IHC_HC(7);
        default: R4D_IHC_HC(8);
    }
#undef R4D_IHC_HC
#undef R4D_IHC
#undef R4D_IHC_K
}

extern "C" int r4d_rad_ihc_post(const RadArgs* a, RadStream s) {
    const RadTensor* y = r4d_opt(a, 0);
    const RadTensor* post = r4d_opt(a, 1);
    const RadTensor* h = r4d_opt(a, 2);
    if (!y || !post || !h) return R4D_NO(RAD_E_INVAL);
    if (!ihc_bf16(y) || !ihc_bf16(h) || !ihc_f32(post)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    if (n < 1 || n > 16384 || hc < 2 || hc > 8) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0, Mp = 0;
    const long long hp = r4d_rows_pitch(h, hc * n, &M), yp = r4d_rows_pitch(y, n, &My),
                    pp = r4d_rows_pitch(post, hc, &Mp);
    if (hp < 0 || yp < 0 || pp < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || My != M || Mp != M) return R4D_NO(RAD_E_SHAPE);
    const bool vec = n % 8 == 0 && ihc_vec(h->data, hp) && ihc_vec(y->data, yp);
    const dim3 grid(vec ? (uint32_t)((n / 8 + 255) / 256) : (uint32_t)((n + 255) / 256), (uint32_t)M);
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;   /* decode heights */
#define R4D_IP_K(V_, C_)                                                                           \
    R4D_EMIT(s, (ihc_post_kernel<V_, C_>), grid, dim3(256), 0, (const uint16_t*)y->data,            \
             (const float*)post->data, (uint16_t*)h->data, (uint32_t)n, (uint32_t)hc,              \
             (uint64_t)yp, (uint64_t)pp, (uint64_t)hp)
    return ch ? (vec ? R4D_IP_K(true, r4dchain) : R4D_IP_K(false, r4dchain))
              : (vec ? R4D_IP_K(true, R4dUnchained) : R4D_IP_K(false, R4dUnchained));
#undef R4D_IP_K
}

extern "C" int r4d_rad_gate_mul(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* g = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !g || !y) return R4D_NO(RAD_E_INVAL);
    if (!ihc_bf16(x) || !ihc_bf16(g) || !ihc_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n < 1) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0, My = 0;
    const long long ap = r4d_rows_pitch(x, n, &M), gp = r4d_rows_pitch(g, n, &Mg),
                    yp = r4d_rows_pitch(y, n, &My);
    if (ap < 0 || gp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || M > 65535 || Mg != M || My != M) return R4D_NO(RAD_E_SHAPE);
    const bool vec = n % 8 == 0 && ihc_vec(x->data, ap) && ihc_vec(g->data, gp) && ihc_vec(y->data, yp);
    const dim3 grid(vec ? (uint32_t)((n / 8 + 255) / 256) : (uint32_t)((n + 255) / 256), (uint32_t)M);
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
#define R4D_GM_K(V_, C_)                                                                           \
    R4D_EMIT(s, (gate_mul_kernel<V_, C_>), grid, dim3(256), 0, (const uint16_t*)x->data,            \
             (const uint16_t*)g->data, (uint16_t*)y->data, (uint32_t)n, (uint64_t)ap,              \
             (uint64_t)gp, (uint64_t)yp)
    return ch ? (vec ? R4D_GM_K(true, r4dchain) : R4D_GM_K(false, r4dchain))
              : (vec ? R4D_GM_K(true, R4dUnchained) : R4D_GM_K(false, R4dUnchained));
#undef R4D_GM_K
}

/* ---- rmsnorm_ihc_junction. Operands (docs/OPS.md):
 *   [0] a:inout  [1] b?  [2] w_post w  [3] post:inout  [4] h:inout  [5] fn w?  [6] scale w?
 *   [7] base w?  [8] w_next w?  [9] xn?  [10] y?  [11] x?
 * `tail` 1 needs fn, scale, base, w_next and xn; `rows` 2 hc (post_mag given) writes the next
 * write weights into `post`, `rows` hc (the head) leaves it. */
enum { IJ_A = 0, IJ_B, IJ_WPOST, IJ_POST, IJ_H, IJ_FN, IJ_SCALE, IJ_BASE, IJ_WNEXT, IJ_XN, IJ_Y, IJ_X,
       IJ_XQ, IJ_XS };

int r4d_ihc_junction_args(const RadArgs* a, r4d_ihc::R4dIhcJunction* o) {
    *o = R4dIhcJunction{};
    const RadTensor* t[14];
    for (int i = 0; i < 14; ++i) t[i] = r4d_opt(a, i);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0),
                    rows = r4d_geti_or(a, "rows", 0), tail = r4d_geti_or(a, "tail", 1);
    if (!t[IJ_A] || !t[IJ_WPOST] || !t[IJ_POST] || !t[IJ_H]) return RAD_E_INVAL;
    if (tail != 0 && tail != 1) return RAD_E_INVAL;
    if (tail && (!t[IJ_FN] || !t[IJ_SCALE] || !t[IJ_BASE] || !t[IJ_WNEXT] || !t[IJ_XN]))
        return RAD_E_INVAL;
    for (int i : { IJ_A, IJ_B, IJ_H, IJ_XN, IJ_Y, IJ_X })
        if (t[i] && !ihc_bf16(t[i])) return RAD_E_DTYPE;
    for (int i : { IJ_WPOST, IJ_POST, IJ_FN, IJ_SCALE, IJ_BASE, IJ_WNEXT })
        if (t[i] && !ihc_f32(t[i])) return RAD_E_DTYPE;
    if (n < 8 || n > 16384 || n % 8 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    /* G-3b: the next norm's rotated int8 codes, 128-column groups (xq [M, n] i8, xs [M, n / 128] f32) */
    const bool codes = t[IJ_XQ] != nullptr;
    if (codes) {
        if (!t[IJ_XS] || !tail || n % 128 || r4d_geti_or(a, "group", 128) != 128) return RAD_E_SHAPE;
        if (t[IJ_XQ]->dtype != RAD_I8 || t[IJ_XS]->dtype != RAD_F32) return RAD_E_DTYPE;
    } else if (t[IJ_XS]) {
        return RAD_E_INVAL;
    }
    const bool post = rows == 2 * hc;
    if (tail && !post && rows != hc) return RAD_E_SHAPE;
    const float eps = (float)r4d_getf(a, "eps", 0), ieps = (float)r4d_getf(a, "ihc_eps", 0),
                heps = (float)r4d_getf(a, "hc_eps", 0);
    if (!std::isfinite(eps) || eps <= 0) return RAD_E_SHAPE;
    float pmag = 0.0f;
    if (tail) {
        if (!std::isfinite(ieps) || ieps <= 0 || !std::isfinite(heps) || heps < 0) return RAD_E_SHAPE;
        if (post) {
            if (r4d_getf(a, "post_mag", -1e30) == -1e30) return RAD_E_INVAL;
            pmag = (float)r4d_getf(a, "post_mag", 0);
            if (!std::isfinite(pmag)) return RAD_E_SHAPE;
        }
    }
    const long long W = hc * n;
    long long M = 0, m2 = 0;
    const long long ap = r4d_rows_pitch(t[IJ_A], n, &M);
    if (ap < 0 || M < 1 || M > 65535) return ap < 0 ? RAD_E_STRIDE : RAD_E_SHAPE;
    auto pitch = [&](int i, long long w, long long* out) {
        if (!t[i]) { *out = 0; return true; }
        *out = r4d_rows_pitch(t[i], w, &m2);
        return *out >= 0 && m2 == M;
    };
    long long bp, pp, hp, xnp = 0, yp, xp;
    if (!pitch(IJ_B, n, &bp) || !pitch(IJ_POST, hc, &pp) || !pitch(IJ_H, W, &hp) ||
        !pitch(IJ_Y, n, &yp) || !pitch(IJ_X, n, &xp) || (tail && !pitch(IJ_XN, n, &xnp)))
        return RAD_E_SHAPE;
    /* whole 16-byte words a row everywhere a row moves as words */
    for (int i : { IJ_A, IJ_B, IJ_H, IJ_XN, IJ_Y, IJ_X }) {
        if (!t[i] || (i == IJ_XN && !tail)) continue;
        long long pv = 0;
        pitch(i, i == IJ_H ? W : n, &pv);
        if (!ihc_vec(t[i]->data, (uint64_t)pv)) return RAD_E_ALIGN;
    }
    long long cqp = 0, csp = 0;
    if (codes) {
        long long mq = 0, ms = 0;
        cqp = r4d_rows_pitch(t[IJ_XQ], n, &mq);
        csp = r4d_rows_pitch(t[IJ_XS], n / 128, &ms);
        if (cqp < 0 || csp < 0 || mq != M || ms != M) return RAD_E_SHAPE;
        if (cqp % 4 || ((uintptr_t)t[IJ_XQ]->data & 3)) return RAD_E_ALIGN;
    }
    if (r4d_numel(t[IJ_WPOST]) != n || !r4d_contig(t[IJ_WPOST]) ||
        ((uintptr_t)t[IJ_WPOST]->data & 15))
        return RAD_E_SHAPE;
    if (tail) {
        const RadTensor* fn = t[IJ_FN];
        if (fn->rank != 2 || fn->shape[0] != rows || fn->shape[1] != W || fn->stride[1] != 1 ||
            fn->stride[0] % 4 || ((uintptr_t)fn->data & 15))
            return RAD_E_SHAPE;
        if (r4d_numel(t[IJ_SCALE]) != (post ? 2 : 1) || r4d_numel(t[IJ_BASE]) != rows ||
            !r4d_contig(t[IJ_SCALE]) || !r4d_contig(t[IJ_BASE]))
            return RAD_E_SHAPE;
        if (r4d_numel(t[IJ_WNEXT]) != n || !r4d_contig(t[IJ_WNEXT]) ||
            ((uintptr_t)t[IJ_WNEXT]->data & 15))
            return RAD_E_SHAPE;
    }
    o->a = (uint16_t*)t[IJ_A]->data;
    o->b = t[IJ_B] ? (const uint16_t*)t[IJ_B]->data : nullptr;
    JuncRow& r = o->r;
    r.n = (uint32_t)n;
    r.eps = eps; r.ieps = ieps; r.heps = heps; r.pmag = pmag; r.wadd = 0.0f; r.mult = 1.0f;
    r.w_post = (const float*)t[IJ_WPOST]->data;
    r.post_row = (float*)t[IJ_POST]->data;
    r.hr = (uint16_t*)t[IJ_H]->data;
    r.fn = tail ? (const float*)t[IJ_FN]->data : nullptr;
    r.fp = tail ? (uint64_t)t[IJ_FN]->stride[0] : 0;
    r.scale = tail ? (const float*)t[IJ_SCALE]->data : nullptr;
    r.base = tail ? (const float*)t[IJ_BASE]->data : nullptr;
    r.w_next = tail ? (const float*)t[IJ_WNEXT]->data : nullptr;
    r.xnr = tail ? (uint16_t*)t[IJ_XN]->data : nullptr;
    r.yr = t[IJ_Y] ? (uint16_t*)t[IJ_Y]->data : nullptr;
    r.xr = t[IJ_X] ? (uint16_t*)t[IJ_X]->data : nullptr;
    r.cq = codes ? (signed char*)t[IJ_XQ]->data : nullptr;
    r.cs = codes ? (float*)t[IJ_XS]->data : nullptr;
    r.cqp = (uint64_t)cqp; r.csp = (uint64_t)csp;
    o->M = M; o->hc = (int)hc; o->post = post; o->tail = tail != 0;
    o->ap = ap; o->bp = bp; o->pp = pp; o->hp = hp; o->xnp = xnp; o->yp = yp; o->xp = xp;
    return RAD_OK;
}

/* Upstream's launch: one row a block (ihc_junction_kernel). */
static int ihc_junction_launch_rows(const R4dIhcJunction& j, RadStream s, bool add, bool ch) {
    const uint32_t lds = j.r.n * 2u;
    const uint32_t grid = r4d_cliff_rows((uint32_t)j.M, 1, kWaves);   /* see ihc_pre_kernel */
#define R4D_IJ_K(HC_, PO_, TA_, AD_, CH_)                                                          \
    R4D_EMIT(s, (ihc_junction_kernel<HC_, PO_, TA_, AD_, CH_>), dim3(grid),                       \
             dim3(kThreads), lds, j.a, j.b, j.r, (uint64_t)j.ap, (uint64_t)j.bp, (uint64_t)j.pp,  \
             (uint64_t)j.hp, (uint64_t)j.xnp, (uint64_t)j.yp, (uint64_t)j.xp, (uint32_t)j.M)
#define R4D_IJ_C(HC_, PO_, TA_, AD_)                                                               \
    (ch ? R4D_IJ_K(HC_, PO_, TA_, AD_, r4dchain) : R4D_IJ_K(HC_, PO_, TA_, AD_, R4dUnchained))
#define R4D_IJ_A(HC_, PO_, TA_) (add ? R4D_IJ_C(HC_, PO_, TA_, true) : R4D_IJ_C(HC_, PO_, TA_, false))
#define R4D_IJ_HC(HC_)                                                                             \
    return !j.tail ? R4D_IJ_A(HC_, false, false)                                                   \
                   : j.post ? R4D_IJ_A(HC_, true, true) : R4D_IJ_A(HC_, false, true)
    switch (j.hc) {
        case 2: R4D_IJ_HC(2);
        case 3: R4D_IJ_HC(3);
        case 4: R4D_IJ_HC(4);
        case 5: R4D_IJ_HC(5);
        case 6: R4D_IJ_HC(6);
        case 7: R4D_IJ_HC(7);
        default: R4D_IJ_HC(8);
    }
#undef R4D_IJ_HC
#undef R4D_IJ_A
#undef R4D_IJ_C
#undef R4D_IJ_K
}

/* The grouped launch over `nbw` working blocks (1..M): the tail at hc 2, 4 and 8 only. */
static int ihc_junction_launch_group(const R4dIhcJunction& j, RadStream s, bool add, bool ch,
                                     uint32_t nbw) {
    constexpr int G = R4D_JUNC_GROUP < 2 ? 2 : R4D_JUNC_GROUP;
    const uint32_t lds = (uint32_t)G * j.r.n * 2u;
    const uint32_t grid = r4d_cliff_rows(nbw, 1, kWaves);   /* see ihc_pre_kernel */
#define R4D_IJG_K(HC_, PO_, AD_, CH_)                                                              \
    R4D_EMIT(s, (ihc_junction_group_kernel<HC_, PO_, AD_, CH_, G>), dim3(grid), dim3(kThreads),  \
             lds, j.a, j.b, j.r, (uint64_t)j.ap, (uint64_t)j.bp, (uint64_t)j.pp, (uint64_t)j.hp,   \
             (uint64_t)j.xnp, (uint64_t)j.yp, (uint64_t)j.xp, (uint32_t)j.M, nbw)
#define R4D_IJG_C(HC_, PO_, AD_)                                                                   \
    (ch ? R4D_IJG_K(HC_, PO_, AD_, r4dchain) : R4D_IJG_K(HC_, PO_, AD_, R4dUnchained))
#define R4D_IJG_A(HC_, PO_) (add ? R4D_IJG_C(HC_, PO_, true) : R4D_IJG_C(HC_, PO_, false))
#define R4D_IJG_HC(HC_) return j.post ? R4D_IJG_A(HC_, true) : R4D_IJG_A(HC_, false)
    switch (j.hc) {
        case 2: R4D_IJG_HC(2);
        case 4: R4D_IJG_HC(4);
        case 8: R4D_IJG_HC(8);
        default: return RAD_E_INVAL;
    }
#undef R4D_IJG_HC
#undef R4D_IJG_A
#undef R4D_IJG_C
#undef R4D_IJG_K
}

/* rmsnorm_ihc_junction with the block count chosen: 0 is the op's (the grouped kernel over
 * ceil(M / G) blocks where junction_grouped says so, else one row a block); -1 forces upstream's
 * one-row kernel; 1..M runs the grouped kernel over that many blocks wherever it is instantiated
 * (tail, even hc, G rows in kJuncLdsBytes), else the one-row kernel. A row's bytes are the same in
 * every case; the card-less tests (tests/r4d_emu/test_hy4.cpp) vary the count. */
extern "C" int r4d_rmsnorm_ihc_junction_nb(const RadArgs* a, RadStream s, long nblocks) {
    R4dIhcJunction j;
    const int rc = r4d_ihc_junction_args(a, &j);
    if (rc != RAD_OK) return R4D_NO(rc);
    const bool add = j.b != nullptr;
    const bool ch = r4d_geti_or(a, "chain", 0) != 0;
    const int g = junction_group(j.hc);
    if (nblocks == 0 && junction_grouped(j.M, j.r.n, j.tail, j.hc))
        nblocks = (long)((j.M + g - 1) / g);
    else if (nblocks > j.M || nblocks < -1) return R4D_NO(RAD_E_INVAL);
    if (nblocks > 0 && junction_grouped(kJuncGroupMinRows, j.r.n, j.tail, j.hc))
        return ihc_junction_launch_group(j, s, add, ch, (uint32_t)nblocks);
    return ihc_junction_launch_rows(j, s, add, ch);
}

extern "C" int r4d_rad_rmsnorm_ihc_junction(const RadArgs* a, RadStream s) {
    return r4d_rmsnorm_ihc_junction_nb(a, s, 0);
}
