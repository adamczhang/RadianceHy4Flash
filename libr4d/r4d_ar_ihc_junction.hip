// r4d_ar_ihc_junction.hip -- ar_rmsnorm_ihc_junction: the exact all-reduce of a block's output with
// Hy4-flash's iHC junction (rmsnorm_ihc_junction, r4d_ihc.h's junction_row) as its epilogue, at two
// ranks and at four (WP16.2). Byte-identical to `[add_bf16 ->] all_reduce (exact) ->
// rmsnorm_ihc_junction`, which is itself the five unfused ops' bytes.
//
//   * A block owns whole rows (the junction's norms close over a row), the partition of
//     ar_rmsnorm_quant_i8 (r4d_rmsnorm_quant_i8.hip), whose transport this copies: at two ranks the
//     one-shot's per-block handshake, the double-buffered slot, the device-resident sequence and the
//     push check (r4d_ar_seq.h), so the instance (r4d_rad_ar_ln_init's, keyed {M_hi * n}) is shared
//     with the standalone all_reduce of the same bound; at four the wide family's one-shot
//     (r4d_ar_wide.h), as ar_rmsnorm_quant_i8_4rank and ar_hc_write_4rank ride it.
//   * With `b` (the MoE join) each block first forms its rows' a = bf16(a + b) and stores it, as
//     add_bf16 does, and that is the message.
//   * The fold is the standalone exact all-reduce's arithmetic: two ranks, the f32 sum of the two
//     bf16 values rounded once as (__hip_bfloat16) rounds (r4d_ar_oneshot_2rank_exact.hip's vadd);
//     four ranks, fp32 from 0.0f in ascending rank, one rounding (fold_word8, as the four-rank
//     one-shot's ar_vsum). The sum is stored back into `a` (the all-reduce is in place) and the
//     junction then reads it.
//   * The exact wire's coded push (two ranks, P2P, from the fused writes' 512 KiB floor, rows of
//     whole 256 values): r4d_ar_arc2.h's slot layout over r4d_ar_arc.h's lossless chunk codec,
//     about 12.1 bits a value on the link, ported with flint-perf affd49b0 (where it carries
//     ar_rmsnorm_had_quant_i8g's exact wire). The peer decodes the very bf16 words the plain push
//     would have carried and folds them in the same arithmetic, so every output is the plain
//     wire's byte for byte. The norms and the calibration read the sum, so the lossy (rotated)
//     wire is never taken here.
//
// The fold reads every source of a word before it stores it, by the thread that pushed it; no peer
// reads this rank's `a`, so in place is safe. 512 threads a block; LDS junction_lds's rows of n
// bf16 (a group's when a block owns several rows, K-46) + the JuncLds, at most about 5 KiB.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include <type_traits>
#include "r4d_chain.h"
#include "r4d_ar_seq.h"
#include "r4d_ar_wide.h"
#include "r4d_ar_arc.h"
#include "r4d_ar_arc2.h"
#include "r4d_ihc.h"
#include "r4d_launch.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>

using namespace r4d_ihc;

namespace {

#define R4D_ARIJ_SPIN_MAX 4000000000ULL

__device__ __forceinline__ void arij_store_rel(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arij_load_acq(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// The two-rank fold of a word: mine + peer in f32, rounded as (__hip_bfloat16) rounds.
__device__ __forceinline__ int4 arij_fold2(const int4 a, const int4 b) {
    const uint32_t wa[4] = { (uint32_t)a.x, (uint32_t)a.y, (uint32_t)a.z, (uint32_t)a.w };
    const uint32_t wb[4] = { (uint32_t)b.x, (uint32_t)b.y, (uint32_t)b.z, (uint32_t)b.w };
    uint32_t o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
        o[k] = bf16_pack_hi(bf16_hi_rne_nan(bf16w_lo(wa[k]) + bf16w_lo(wb[k])),
                            bf16_hi_rne_nan(bf16w_hi(wa[k]) + bf16w_hi(wb[k])));
    return make_int4((int)o[0], (int)o[1], (int)o[2], (int)o[3]);
}

// The four-rank fold of a word (fold_word8, r4d_rmsnorm_quant_i8.hip): fp32 from 0.0f, the sources
// in ascending rank, one rounding through bf16_hi_rne_nan.
template <int WS>
__device__ __forceinline__ int4 arij_foldN(const int4* const* src4, size_t k) {
    int4 v[WS];
#pragma unroll
    for (int r = 0; r < WS; ++r) v[r] = src4[r][k];
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
        const uint16_t* pv = reinterpret_cast<const uint16_t*>(&v[r]);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += bf16_to_f32(pv[j]);
    }
    int4 a;
    uint16_t* pa = reinterpret_cast<uint16_t*>(&a);
#pragma unroll
    for (int j = 0; j < 8; ++j) pa[j] = (uint16_t)(bf16_hi_rne_nan(acc[j]) >> 16);
    return a;
}

// add_bf16's word: bf16(f32(a) + f32(b)).
__device__ __forceinline__ int4 arij_add(const int4 a, const uint16_t* b) {
    float u[8], v[8];
    unpack8(make_uint4((uint32_t)a.x, (uint32_t)a.y, (uint32_t)a.z, (uint32_t)a.w), u);
    unpack8(*(const uint4*)b, v);
    uint16_t o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) o[e] = f32_to_bf16(u[e] + v[e]);
    const uint4 w = pack8(o);
    return make_int4((int)w.x, (int)w.y, (int)w.z, (int)w.w);
}

// The rows [r0, r1) of `a`, reduced, through the junction (r4d_ihc.h's junction_walk: groups of
// junction_tile rows sharing their fn reads, K-46, then one by one; a row's bytes are the same).
template <int HC, bool POST, bool TAIL>
__device__ __forceinline__ void arij_rows(JuncLds<HC, POST>* L, uint16_t* ybuf, uint16_t* a,
                                          const JuncRow& r, int r0, int r1, uint64_t pp, uint64_t hp,
                                          uint64_t xnp, uint64_t yp, uint64_t xp) {
    const uint32_t wpr = r.n / 8u;
    junction_walk<HC, POST, TAIL>(L, ybuf, r, r0, r1, pp, hp, xnp, yp, xp,
                                  [&](int row, uint16_t* buf) {
        const uint4* ar = (const uint4*)(a + (size_t)row * r.n);
        for (uint32_t q = threadIdx.x; q < wpr; q += kThreads) ((uint4*)buf)[q] = ar[q];
    });
}

#define R4D_ARIJ_ROWPARAMS                                                                         \
    uint16_t* __restrict__ a, const uint16_t* __restrict__ b, JuncRow r, uint64_t bp, uint64_t pp, \
    uint64_t hp, uint64_t xnp, uint64_t yp, uint64_t xp, int64_t M

// ---- two ranks. CODED: the push coded (r4d_ar_arc2.h); the launcher takes it for rows of whole
// 256 values, so a block's word range is whole waves' and every lane of a wave is in each pass.
template <int HC, bool POST, bool TAIL, bool ADD, bool CODED, class CH>
__global__ __launch_bounds__(kThreads) void ar_ihc_junction_2rank_kernel(
    R4D_ARIJ_ROWPARAMS, int4* peer_base, const int4* my_base, unsigned int* peer_flags,
    unsigned int* my_flags, unsigned int* seq_ctrs, int slot_stride16, int drain, int acq,
    unsigned int nbw, long stage) {
    if (CH::on) r4d_chain_wait();
    R4D_DYN_LDS(uint4, jbuf);
    __shared__ JuncLds<HC, POST> L[TAIL ? junction_group(HC) : 1];
    __shared__ unsigned int s_seq, s_slot;
    const int bk = (int)blockIdx.x, nb = (int)gridDim.x;
    const int tid = (int)threadIdx.x, lane = tid & 31;
    const int r0 = (int)(((long)bk * M) / nb), r1 = (int)(((long)(bk + 1) * M) / nb);
    const int wpr = (int)(r.n >> 3);
    const int k0 = r0 * wpr, k1 = r1 * wpr;
    int4* const a4 = (int4*)a;

    r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int sq = s_seq;
    int4* const ps = peer_base + (size_t)s_slot * slot_stride16;
    const int4* ms = my_base + (size_t)s_slot * slot_stride16;
    const long md = CODED ? 0 : r4d_archk_md(stage, my_base);   /* the host mailbox's mirror */
    const int n16 = CODED ? M * wpr : 0;

    /* the message: my rows (after the add), pushed */
    unsigned int hck = 0u;
    for (int k = k0 + tid; k < k1; k += kThreads) {
        int4 v = a4[k];
        if (ADD) {
            const int row = k / wpr, q = k - row * wpr;
            v = arij_add(v, b + (size_t)row * bp + 8 * q);
            a4[k] = v;
        }
        if (CODED) {
            hck += r4d_arc2_put(r4d_arc2_at(ps, n16), k - lane, v);
        } else {
            ps[k] = v;
            hck += r4d_archk_u4(v, 4u * (unsigned)k);
        }
    }
    r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)bk, sq);
    if (drain == 1) __threadfence_system();
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (drain == 2) __threadfence_system();
        arij_store_rel(&peer_flags[bk], sq);
        unsigned long long z = 0, t0 = 0ull;
        while (r4d_ar_seq_behind(arij_load_acq(&my_flags[bk]), sq))
            if (++z > R4D_ARIJ_SPIN_MAX || r4d_ar_spin_late(z, t0)) __builtin_trap();
    }
    __syncthreads();
    if (acq) __threadfence_system();
    r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)bk, sq, [&](unsigned int& h) {
        for (int k = k0 + tid; k < k1; k += kThreads) {
            if (CODED) h += r4d_arc2_sum(r4d_arc2_at(ms, n16), k - lane);
            else h += r4d_archk_u4(r4d_archk_keep(&ms[k], md), 4u * (unsigned)k);
        }
    });
    if (!CODED) ms = r4d_archk_moved(ms, md);

    /* the fold into `a`, each thread its own words */
    for (int k = k0 + tid; k < k1; k += kThreads) {
        const int4 pv = CODED ? r4d_arc2_get(r4d_arc2_at(ms, n16), k - lane) : ms[k];
        a4[k] = arij_fold2(a4[k], pv);
    }
    __syncthreads();
    arij_rows<HC, POST, TAIL>(L, (uint16_t*)jbuf, a, r, r0, r1, pp, hp, xnp, yp, xp);
    if (CH::on) r4d_chain_done();
}

// ---- four ranks: the wide family's exact one-shot (r4d_ar_oneshot_Nrank_exact.hip), as
// ar_rmsnorm_quant_i8_wide_kernel rides it, with the junction as its epilogue.
// The peers as a kernel argument without the padding R4DArWidePeers<int4, 3> has behind its three
// ranks (a recorded pass's audit compares argument bytes); rebuilt as one in the kernel.
struct ArijPeers {
    int4* scratch[3];
    unsigned int* flags[3];
    int rank[4];
};
static_assert(sizeof(ArijPeers) == sizeof(R4DArWidePeers<int4, 3>) &&
                  std::has_unique_object_representations_v<ArijPeers>,
              "ArijPeers is a kernel argument by value and carries no padding");
template <int HC, bool POST, bool TAIL, bool ADD, class CH>
__global__ __launch_bounds__(kThreads) void ar_ihc_junction_4rank_kernel(
    R4D_ARIJ_ROWPARAMS, ArijPeers pa, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs, int slot_stride16,
    int rank, int drain, int acq, int pub) {
    constexpr int WS = 4, NPEERS = WS - 1;
    if (CH::on) r4d_chain_wait();
    R4DArWidePeers<int4, NPEERS> peers;
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) {
        peers.scratch[p] = pa.scratch[p];
        peers.flags[p] = pa.flags[p];
        peers.rank[p] = pa.rank[p];
    }
    R4D_DYN_LDS(uint4, jbuf);
    __shared__ JuncLds<HC, POST> L[TAIL ? junction_group(HC) : 1];
    __shared__ unsigned int s_seq, s_slot;
    const int bk = (int)blockIdx.x, nb = (int)gridDim.x, tid = (int)threadIdx.x;
    const int r0 = (int)(((long)bk * M) / nb), r1 = (int)(((long)(bk + 1) * M) / nb);
    const int wpr = (int)(r.n >> 3);
    const int k0 = r0 * wpr, k1 = r1 * wpr;
    int4* const a4 = (int4*)a;

    r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
    const unsigned int s = s_seq;
    const int parity = (int)s_slot;
    int4* ps[NPEERS];
#pragma unroll
    for (int p = 0; p < NPEERS; ++p)
        ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
    for (int k = k0 + tid; k < k1; k += kThreads) {
        int4 v = a4[k];
        if (ADD) {
            const int row = k / wpr, q = k - row * wpr;
            v = arij_add(v, b + (size_t)row * bp + 8 * q);
            a4[k] = v;
        }
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }
    r4d_ar_handshake<WS>(peers, my_flags, bk, rank, 2u * s, drain, acq, pub);

    const int4* src4[WS];
#pragma unroll
    for (int q = 0; q < WS; ++q)
        src4[q] = q == rank ? (const int4*)a4
                            : my_base + ((size_t)parity * WS + q) * (size_t)slot_stride16;
    for (int k = k0 + tid; k < k1; k += kThreads) a4[k] = arij_foldN<WS>(src4, (size_t)k);
    __syncthreads();
    arij_rows<HC, POST, TAIL>(L, (uint16_t*)jbuf, a, r, r0, r1, pp, hp, xnp, yp, xp);
    if (CH::on) r4d_chain_done();
}

// A block count off the full-card cliff (r4d_cliff_rows): one block fewer where the grid's waves
// would land on it. Blocks own whole rows by an even split, so the bytes do not depend on the count,
// and every rank derives the same count from the same M.
inline int arij_off_cliff(int nb) {
    while (nb > 1 && r4d_cliff_rows((uint32_t)nb, 1, kWaves) != (uint32_t)nb) --nb;
    return nb;
}

}  // namespace

// The two-rank launch: the IPC arguments are r4d_ar_rmsnorm_quant_i8's, `j` the checked operands
// (r4d_ihc_junction_args), `coded` the exact wire's coded push (the shim decides it from the
// floor and r4d_ar_coded2_ok; it is taken only for n % 256 == 0). Returns 0 or a negative code.
extern "C" int r4d_ar_ihc_junction_2rank(long peer_scratch, long my_scratch, long peer_flags,
                                         long my_flags, long seq_ctrs, long slot_stride16,
                                         const void* jv, long stream, int nblocks, int drain,
                                         int acq, long stage, int chain, int coded) {
    const R4dIhcJunction& j = *(const R4dIhcJunction*)jv;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs) return -1;
    if (j.ap != (long long)j.r.n) return -2;                         /* the message is tight */
    if ((long)j.M * j.r.n / 8 > slot_stride16) return -7;
    if (coded && (j.r.n % 256 || stage)) return -3;
    const int M = (int)j.M;
    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    const int nbw = r4d_ar_max_blocks();
    if (nb > nbw) nb = nbw;
    nb = arij_off_cliff(nb);
    const bool ch = chain != 0;
    const bool add = j.b != nullptr;
    const uint32_t lds = junction_lds(j.r.n, j.tail, j.hc, M, nb);
#define R4D_ARIJ2_K(HC_, PO_, TA_, AD_, CO_, CH_)                                                  \
    R4D_EMIT((RadStream)stream, (ar_ihc_junction_2rank_kernel<HC_, PO_, TA_, AD_, CO_, CH_>),     \
             dim3((uint32_t)nb), dim3(kThreads), lds, j.a, j.b, j.r, (uint64_t)j.bp,               \
             (uint64_t)j.pp, (uint64_t)j.hp, (uint64_t)j.xnp, (uint64_t)j.yp, (uint64_t)j.xp,       \
             (int64_t)M, (int4*)peer_scratch, (const int4*)my_scratch, (unsigned int*)peer_flags,              \
             (unsigned int*)my_flags, (unsigned int*)seq_ctrs, (int)slot_stride16, drain, acq,    \
             (unsigned)nbw, stage)
#define R4D_ARIJ2_C(HC_, PO_, TA_, AD_)                                                            \
    (coded ? (ch ? R4D_ARIJ2_K(HC_, PO_, TA_, AD_, true, r4dchain)                                 \
                 : R4D_ARIJ2_K(HC_, PO_, TA_, AD_, true, R4dUnchained))                            \
           : (ch ? R4D_ARIJ2_K(HC_, PO_, TA_, AD_, false, r4dchain)                                \
                 : R4D_ARIJ2_K(HC_, PO_, TA_, AD_, false, R4dUnchained)))
#define R4D_ARIJ2_A(HC_, PO_, TA_)                                                                 \
    (add ? R4D_ARIJ2_C(HC_, PO_, TA_, true) : R4D_ARIJ2_C(HC_, PO_, TA_, false))
#define R4D_ARIJ2_HC(HC_)                                                                          \
    rc = !j.tail ? R4D_ARIJ2_A(HC_, false, false)                                                  \
                 : j.post ? R4D_ARIJ2_A(HC_, true, true) : R4D_ARIJ2_A(HC_, false, true)
    int rc;
    switch (j.hc) {
        case 2: R4D_ARIJ2_HC(2); break;
        case 4: R4D_ARIJ2_HC(4); break;
        case 8: R4D_ARIJ2_HC(8); break;
        default: return -5;
    }
#undef R4D_ARIJ2_HC
#undef R4D_ARIJ2_A
#undef R4D_ARIJ2_C
#undef R4D_ARIJ2_K
    return rc == RAD_OK ? 0 : -4;
}

// The four-rank launch: r4d_ar_rmsnorm_quant_i8_4rank's IPC arguments and checks, one-shot only.
extern "C" int r4d_ar_ihc_junction_4rank(long peer_scratch0, long peer_scratch1,
                                         long peer_scratch2, long peer_flags0, long peer_flags1,
                                         long peer_flags2, long my_scratch, long my_flags,
                                         long seq_ctrs, long slot_stride16, const void* jv,
                                         int rank, long stream, int nblocks, int drain, int acq,
                                         int pub, int chain) {
    constexpr int WS = 4, NPEERS = WS - 1;
    const R4dIhcJunction& j = *(const R4dIhcJunction*)jv;
    const long ps[NPEERS] = { peer_scratch0, peer_scratch1, peer_scratch2 };
    const long pf[NPEERS] = { peer_flags0, peer_flags1, peer_flags2 };
    if (!my_scratch || !my_flags || !seq_ctrs) return -1;
    if (rank < 0 || rank >= WS) return -1;
    if (j.ap != (long long)j.r.n) return -2;
    if ((long)j.M * j.r.n / 8 > slot_stride16) return -7;
    const int M = (int)j.M;
    ArijPeers peers;
    std::memset(&peers, 0, sizeof peers);
    for (int i = 0; i < NPEERS; ++i) {
        if (!ps[i] || !pf[i]) return -1;
        peers.scratch[i] = (int4*)ps[i];
        peers.flags[i] = (unsigned int*)pf[i];
        peers.rank[i] = i < rank ? i : i + 1;
    }
    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
    nb = arij_off_cliff(nb);
    try {
        const long slot_bytes = 2L * WS * slot_stride16 * 16;
        const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
        r4d_ar_check_extent(my_scratch, slot_bytes, "ar_rmsnorm_ihc_junction", "my_scratch");
        r4d_ar_check_extent(my_flags, flag_bytes, "ar_rmsnorm_ihc_junction", "my_flags");
        r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                            "ar_rmsnorm_ihc_junction", "seq_ctrs");
        for (int i = 0; i < NPEERS; ++i) {
            r4d_ar_check_extent(ps[i], slot_bytes, "ar_rmsnorm_ihc_junction", "peer scratch");
            r4d_ar_check_extent(pf[i], flag_bytes, "ar_rmsnorm_ihc_junction", "peer flags");
        }
    } catch (...) {
        return -8;
    }
    const bool ch = chain != 0;
    const bool add = j.b != nullptr;
    const uint32_t lds = junction_lds(j.r.n, j.tail, j.hc, M, nb);
#define R4D_ARIJ4_K(HC_, PO_, TA_, AD_, CH_)                                                       \
    R4D_EMIT((RadStream)stream, (ar_ihc_junction_4rank_kernel<HC_, PO_, TA_, AD_, CH_>),          \
             dim3((uint32_t)nb), dim3(kThreads), lds, j.a, j.b, j.r, (uint64_t)j.bp,               \
             (uint64_t)j.pp, (uint64_t)j.hp, (uint64_t)j.xnp, (uint64_t)j.yp, (uint64_t)j.xp,       \
             (int64_t)M, peers, (const int4*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs,     \
             (int)slot_stride16, rank, drain, acq, pub)
#define R4D_ARIJ4_C(HC_, PO_, TA_, AD_)                                                            \
    (ch ? R4D_ARIJ4_K(HC_, PO_, TA_, AD_, r4dchain) : R4D_ARIJ4_K(HC_, PO_, TA_, AD_, R4dUnchained))
#define R4D_ARIJ4_A(HC_, PO_, TA_)                                                                 \
    (add ? R4D_ARIJ4_C(HC_, PO_, TA_, true) : R4D_ARIJ4_C(HC_, PO_, TA_, false))
#define R4D_ARIJ4_HC(HC_)                                                                          \
    rc = !j.tail ? R4D_ARIJ4_A(HC_, false, false)                                                  \
                 : j.post ? R4D_ARIJ4_A(HC_, true, true) : R4D_ARIJ4_A(HC_, false, true)
    int rc;
    switch (j.hc) {
        case 2: R4D_ARIJ4_HC(2); break;
        case 4: R4D_ARIJ4_HC(4); break;
        case 8: R4D_ARIJ4_HC(8); break;
        default: return -5;
    }
#undef R4D_ARIJ4_HC
#undef R4D_ARIJ4_A
#undef R4D_ARIJ4_C
#undef R4D_ARIJ4_K
    return rc == RAD_OK ? 0 : -4;
}
