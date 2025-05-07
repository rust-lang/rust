// We're testing x86 target specific features
//@only-target: x86_64 i686
#![allow(unnecessary_transmutes)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::f32::NAN;
use std::mem::transmute;

fn main() {
    assert!(is_x86_feature_detected!("sse"));

    unsafe {
        test_sse();
    }
}

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < $eps,
            "assertion failed: `(left !== right)` \
             (left: `{:?}`, right: `{:?}`, expect diff: `{:?}`, real diff: `{:?}`)",
            *a,
            *b,
            $eps,
            (*a - *b).abs()
        );
    }};
}

#[target_feature(enable = "sse")]
unsafe fn test_sse() {
    // Mostly copied from library/stdarch/crates/core_arch/src/x86{,_64}/sse.rs

    #[target_feature(enable = "sse")]
    unsafe fn assert_eq_m128(a: __m128, b: __m128) {
        let r = _mm_cmpeq_ps(a, b);
        if _mm_movemask_ps(r) != 0b1111 {
            panic!("{:?} != {:?}", a, b);
        }
    }

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_add_ss() {
        let a = _mm_set_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_set_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_add_ss(a, b);
        assert_eq_m128(r, _mm_set_ps(-1.0, 5.0, 0.0, -15.0));
    }
    test_mm_add_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_sub_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_sub_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(99.0, 5.0, 0.0, -10.0));
    }
    test_mm_sub_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_mul_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_mul_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(100.0, 5.0, 0.0, -10.0));
    }
    test_mm_mul_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_div_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_div_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(0.01, 5.0, 0.0, -10.0));
    }
    test_mm_div_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_sqrt_ss() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_sqrt_ss(a);
        let e = _mm_setr_ps(2.0, 13.0, 16.0, 100.0);
        assert_eq_m128(r, e);
    }
    test_mm_sqrt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_sqrt_ps() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_sqrt_ps(a);
        let e = _mm_setr_ps(2.0, 3.6055512, 4.0, 10.0);
        assert_eq_m128(r, e);
    }
    test_mm_sqrt_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_rcp_ss() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rcp_ss(a);
        let e = _mm_setr_ps(0.24993896, 13.0, 16.0, 100.0);
        let rel_err = 0.00048828125;

        let r: [f32; 4] = transmute(r);
        let e: [f32; 4] = transmute(e);
        assert_approx_eq!(r[0], e[0], 2. * rel_err);
        for i in 1..4 {
            assert_eq!(r[i], e[i]);
        }
    }
    test_mm_rcp_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_rcp_ps() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rcp_ps(a);
        let e = _mm_setr_ps(0.24993896, 0.0769043, 0.06248474, 0.0099983215);
        let rel_err = 0.00048828125;

        let r: [f32; 4] = transmute(r);
        let e: [f32; 4] = transmute(e);
        for i in 0..4 {
            assert_approx_eq!(r[i], e[i], 2. * rel_err);
        }
    }
    test_mm_rcp_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_rsqrt_ss() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rsqrt_ss(a);
        let e = _mm_setr_ps(0.49987793, 13.0, 16.0, 100.0);
        let rel_err = 0.00048828125;

        let r: [f32; 4] = transmute(r);
        let e: [f32; 4] = transmute(e);
        assert_approx_eq!(r[0], e[0], 2. * rel_err);
        for i in 1..4 {
            assert_eq!(r[i], e[i]);
        }
    }
    test_mm_rsqrt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_rsqrt_ps() {
        let a = _mm_setr_ps(4.0, 13.0, 16.0, 100.0);
        let r = _mm_rsqrt_ps(a);
        let e = _mm_setr_ps(0.49987793, 0.2772827, 0.24993896, 0.099990845);
        let rel_err = 0.00048828125;

        let r: [f32; 4] = transmute(r);
        let e: [f32; 4] = transmute(e);
        for i in 0..4 {
            assert_approx_eq!(r[i], e[i], 2. * rel_err);
        }
    }
    test_mm_rsqrt_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_min_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_min_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(-100.0, 5.0, 0.0, -10.0));
    }
    test_mm_min_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_min_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_min_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-100.0, 5.0, 0.0, -10.0));

        // `_mm_min_ps` can **not** be implemented using the `simd_min` rust intrinsic because
        // the semantics of `simd_min` are different to those of `_mm_min_ps` regarding handling
        // of `-0.0`.
        let a = _mm_setr_ps(-0.0, 0.0, 0.0, 0.0);
        let b = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);
        let r1: [u8; 16] = transmute(_mm_min_ps(a, b));
        let r2: [u8; 16] = transmute(_mm_min_ps(b, a));
        let a: [u8; 16] = transmute(a);
        let b: [u8; 16] = transmute(b);
        assert_eq!(r1, b);
        assert_eq!(r2, a);
        assert_ne!(a, b); // sanity check that -0.0 is actually present
    }
    test_mm_min_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_max_ss() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_max_ss(a, b);
        assert_eq_m128(r, _mm_setr_ps(-1.0, 5.0, 0.0, -10.0));
    }
    test_mm_max_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_max_ps() {
        let a = _mm_setr_ps(-1.0, 5.0, 0.0, -10.0);
        let b = _mm_setr_ps(-100.0, 20.0, 0.0, -5.0);
        let r = _mm_max_ps(a, b);
        assert_eq_m128(r, _mm_setr_ps(-1.0, 20.0, 0.0, -5.0));

        // `_mm_max_ps` can **not** be implemented using the `simd_max` rust intrinsic because
        // the semantics of `simd_max` are different to those of `_mm_max_ps` regarding handling
        // of `-0.0`.
        let a = _mm_setr_ps(-0.0, 0.0, 0.0, 0.0);
        let b = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);
        let r1: [u8; 16] = transmute(_mm_max_ps(a, b));
        let r2: [u8; 16] = transmute(_mm_max_ps(b, a));
        let a: [u8; 16] = transmute(a);
        let b: [u8; 16] = transmute(b);
        assert_eq!(r1, b);
        assert_eq!(r2, a);
        assert_ne!(a, b); // sanity check that -0.0 is actually present
    }
    test_mm_max_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpeq_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(-1.0, 5.0, 6.0, 7.0);
        let r: [u32; 4] = transmute(_mm_cmpeq_ss(a, b));
        let e: [u32; 4] = transmute(_mm_setr_ps(transmute(0u32), 2.0, 3.0, 4.0));
        assert_eq!(r, e);

        let b2 = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let r2: [u32; 4] = transmute(_mm_cmpeq_ss(a, b2));
        let e2: [u32; 4] = transmute(_mm_setr_ps(transmute(0xffffffffu32), 2.0, 3.0, 4.0));
        assert_eq!(r2, e2);
    }
    test_mm_cmpeq_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmplt_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) < b.extract(0)
        let c1 = 0u32; // a.extract(0) < c.extract(0)
        let d1 = !0u32; // a.extract(0) < d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmplt_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmplt_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmplt_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmplt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmple_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) <= b.extract(0)
        let c1 = !0u32; // a.extract(0) <= c.extract(0)
        let d1 = !0u32; // a.extract(0) <= d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmple_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmple_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmple_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmple_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpgt_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) > b.extract(0)
        let c1 = 0u32; // a.extract(0) > c.extract(0)
        let d1 = 0u32; // a.extract(0) > d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpgt_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpgt_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpgt_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpgt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpge_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) >= b.extract(0)
        let c1 = !0u32; // a.extract(0) >= c.extract(0)
        let d1 = 0u32; // a.extract(0) >= d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpge_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpge_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpge_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpge_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpneq_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) != b.extract(0)
        let c1 = 0u32; // a.extract(0) != c.extract(0)
        let d1 = !0u32; // a.extract(0) != d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpneq_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpneq_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpneq_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpneq_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpnlt_ss() {
        // TODO: this test is exactly the same as for `_mm_cmpge_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence of NaNs (signaling or quiet). If so, we should add tests
        // for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) >= b.extract(0)
        let c1 = !0u32; // a.extract(0) >= c.extract(0)
        let d1 = 0u32; // a.extract(0) >= d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpnlt_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpnlt_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpnlt_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpnlt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpnle_ss() {
        // TODO: this test is exactly the same as for `_mm_cmpgt_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence
        // of NaNs (signaling or quiet). If so, we should add tests for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) > b.extract(0)
        let c1 = 0u32; // a.extract(0) > c.extract(0)
        let d1 = 0u32; // a.extract(0) > d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpnle_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpnle_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpnle_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpnle_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpngt_ss() {
        // TODO: this test is exactly the same as for `_mm_cmple_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence of NaNs (signaling or quiet). If so, we should add tests
        // for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) <= b.extract(0)
        let c1 = !0u32; // a.extract(0) <= c.extract(0)
        let d1 = !0u32; // a.extract(0) <= d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpngt_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpngt_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpngt_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpngt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpnge_ss() {
        // TODO: this test is exactly the same as for `_mm_cmplt_ss`, but there
        // must be a difference. It may have to do with behavior in the
        // presence of NaNs (signaling or quiet). If so, we should add tests
        // for those.

        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(1.0, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) < b.extract(0)
        let c1 = 0u32; // a.extract(0) < c.extract(0)
        let d1 = !0u32; // a.extract(0) < d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpnge_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpnge_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpnge_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpnge_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpord_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(NAN, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = !0u32; // a.extract(0) ord b.extract(0)
        let c1 = 0u32; // a.extract(0) ord c.extract(0)
        let d1 = !0u32; // a.extract(0) ord d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpord_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpord_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpord_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpord_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpunord_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(0.0, 5.0, 6.0, 7.0);
        let c = _mm_setr_ps(NAN, 5.0, 6.0, 7.0);
        let d = _mm_setr_ps(2.0, 5.0, 6.0, 7.0);

        let b1 = 0u32; // a.extract(0) unord b.extract(0)
        let c1 = !0u32; // a.extract(0) unord c.extract(0)
        let d1 = 0u32; // a.extract(0) unord d.extract(0)

        let rb: [u32; 4] = transmute(_mm_cmpunord_ss(a, b));
        let eb: [u32; 4] = transmute(_mm_setr_ps(transmute(b1), 2.0, 3.0, 4.0));
        assert_eq!(rb, eb);

        let rc: [u32; 4] = transmute(_mm_cmpunord_ss(a, c));
        let ec: [u32; 4] = transmute(_mm_setr_ps(transmute(c1), 2.0, 3.0, 4.0));
        assert_eq!(rc, ec);

        let rd: [u32; 4] = transmute(_mm_cmpunord_ss(a, d));
        let ed: [u32; 4] = transmute(_mm_setr_ps(transmute(d1), 2.0, 3.0, 4.0));
        assert_eq!(rd, ed);
    }
    test_mm_cmpunord_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpeq_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = [fls, fls, tru, fls];
        let r: [u32; 4] = transmute(_mm_cmpeq_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpeq_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmplt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = [tru, fls, fls, fls];
        let r: [u32; 4] = transmute(_mm_cmplt_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmplt_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmple_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, 4.0);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = [tru, fls, tru, fls];
        let r: [u32; 4] = transmute(_mm_cmple_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmple_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpgt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 42.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = [fls, tru, fls, fls];
        let r: [u32; 4] = transmute(_mm_cmpgt_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpgt_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpge_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 42.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = [fls, tru, tru, fls];
        let r: [u32; 4] = transmute(_mm_cmpge_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpge_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpneq_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = [tru, tru, fls, tru];
        let r: [u32; 4] = transmute(_mm_cmpneq_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpneq_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpnlt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = [fls, tru, tru, tru];
        let r: [u32; 4] = transmute(_mm_cmpnlt_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpnlt_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpnle_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = [fls, tru, fls, tru];
        let r: [u32; 4] = transmute(_mm_cmpnle_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpnle_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpngt_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = [tru, fls, tru, tru];
        let r: [u32; 4] = transmute(_mm_cmpngt_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpngt_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpnge_ps() {
        let a = _mm_setr_ps(10.0, 50.0, 1.0, NAN);
        let b = _mm_setr_ps(15.0, 20.0, 1.0, 5.0);
        let tru = !0u32;
        let fls = 0u32;

        let e = [tru, fls, fls, tru];
        let r: [u32; 4] = transmute(_mm_cmpnge_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpnge_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpord_ps() {
        let a = _mm_setr_ps(10.0, 50.0, NAN, NAN);
        let b = _mm_setr_ps(15.0, NAN, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = [tru, fls, fls, fls];
        let r: [u32; 4] = transmute(_mm_cmpord_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpord_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cmpunord_ps() {
        let a = _mm_setr_ps(10.0, 50.0, NAN, NAN);
        let b = _mm_setr_ps(15.0, NAN, 1.0, NAN);
        let tru = !0u32;
        let fls = 0u32;

        let e = [fls, tru, tru, tru];
        let r: [u32; 4] = transmute(_mm_cmpunord_ps(a, b));
        assert_eq!(r, e);
    }
    test_mm_cmpunord_ps();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_comieq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comieq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comieq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_comieq_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_comilt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comilt_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comilt_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_comilt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_comile_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comile_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comile_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_comile_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_comigt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 1, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comige_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comige_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_comigt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_comineq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 1, 1];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_comineq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_comineq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_comineq_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_ucomieq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomieq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomieq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_ucomieq_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_ucomilt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomilt_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomilt_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_ucomilt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_ucomile_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 1, 0, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomile_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomile_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_ucomile_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_ucomigt_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 0, 1, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomigt_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomigt_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_ucomigt_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_ucomige_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[1i32, 0, 1, 0];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomige_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomige_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_ucomige_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_ucomineq_ss() {
        let aa = &[3.0f32, 12.0, 23.0, NAN];
        let bb = &[3.0f32, 47.5, 1.5, NAN];

        let ee = &[0i32, 1, 1, 1];

        for i in 0..4 {
            let a = _mm_setr_ps(aa[i], 1.0, 2.0, 3.0);
            let b = _mm_setr_ps(bb[i], 0.0, 2.0, 4.0);

            let r = _mm_ucomineq_ss(a, b);

            assert_eq!(
                ee[i], r,
                "_mm_ucomineq_ss({:?}, {:?}) = {}, expected: {} (i={})",
                a, b, r, ee[i], i
            );
        }
    }
    test_mm_ucomineq_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvtss_si32() {
        let inputs = &[42.0f32, -3.1, 4.0e10, 4.0e-20, NAN, 2147483500.1];
        let result = &[42i32, -3, i32::MIN, 0, i32::MIN, 2147483520];
        for i in 0..inputs.len() {
            let x = _mm_setr_ps(inputs[i], 1.0, 3.0, 4.0);
            let e = result[i];
            let r = _mm_cvtss_si32(x);
            assert_eq!(e, r, "TestCase #{} _mm_cvtss_si32({:?}) = {}, expected: {}", i, x, r, e);
        }
    }
    test_mm_cvtss_si32();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvttss_si32() {
        let inputs = &[
            (42.0f32, 42i32),
            (-31.4, -31),
            (-33.5, -33),
            (-34.5, -34),
            (10.999, 10),
            (-5.99, -5),
            (4.0e10, i32::MIN),
            (4.0e-10, 0),
            (NAN, i32::MIN),
            (2147483500.1, 2147483520),
        ];
        for i in 0..inputs.len() {
            let (xi, e) = inputs[i];
            let x = _mm_setr_ps(xi, 1.0, 3.0, 4.0);
            let r = _mm_cvttss_si32(x);
            assert_eq!(e, r, "TestCase #{} _mm_cvttss_si32({:?}) = {}, expected: {}", i, x, r, e);
        }
    }
    test_mm_cvttss_si32();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvtss_f32() {
        let a = _mm_setr_ps(312.0134, 5.0, 6.0, 7.0);
        assert_eq!(_mm_cvtss_f32(a), 312.0134);
    }
    test_mm_cvtss_f32();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvtsi32_ss() {
        let inputs = &[
            (4555i32, 4555.0f32),
            (322223333, 322223330.0),
            (-432, -432.0),
            (-322223333, -322223330.0),
        ];

        for i in 0..inputs.len() {
            let (x, f) = inputs[i];
            let a = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
            let r = _mm_cvtsi32_ss(a, x);
            let e = _mm_setr_ps(f, 6.0, 7.0, 8.0);
            assert_eq_m128(e, r);
        }
    }
    test_mm_cvtsi32_ss();

    // Intrinsic only available on x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvtss_si64() {
        let inputs = &[
            (42.0f32, 42i64),
            (-31.4, -31),
            (-33.5, -34),
            (-34.5, -34),
            (4.0e10, 40_000_000_000),
            (4.0e-10, 0),
            (f32::NAN, i64::MIN),
            (2147483500.1, 2147483520),
            (9.223371e18, 9223370937343148032),
        ];
        for i in 0..inputs.len() {
            let (xi, e) = inputs[i];
            let x = _mm_setr_ps(xi, 1.0, 3.0, 4.0);
            let r = _mm_cvtss_si64(x);
            assert_eq!(e, r, "TestCase #{} _mm_cvtss_si64({:?}) = {}, expected: {}", i, x, r, e);
        }
    }
    #[cfg(target_arch = "x86_64")]
    test_mm_cvtss_si64();

    // Intrinsic only available on x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvttss_si64() {
        let inputs = &[
            (42.0f32, 42i64),
            (-31.4, -31),
            (-33.5, -33),
            (-34.5, -34),
            (10.999, 10),
            (-5.99, -5),
            (4.0e10, 40_000_000_000),
            (4.0e-10, 0),
            (f32::NAN, i64::MIN),
            (2147483500.1, 2147483520),
            (9.223371e18, 9223370937343148032),
            (9.223372e18, i64::MIN),
        ];
        for i in 0..inputs.len() {
            let (xi, e) = inputs[i];
            let x = _mm_setr_ps(xi, 1.0, 3.0, 4.0);
            let r = _mm_cvttss_si64(x);
            assert_eq!(e, r, "TestCase #{} _mm_cvttss_si64({:?}) = {}, expected: {}", i, x, r, e);
        }
    }
    #[cfg(target_arch = "x86_64")]
    test_mm_cvttss_si64();

    // Intrinsic only available on x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse")]
    unsafe fn test_mm_cvtsi64_ss() {
        let inputs = &[
            (4555i64, 4555.0f32),
            (322223333, 322223330.0),
            (-432, -432.0),
            (-322223333, -322223330.0),
            (9223372036854775807, 9.223372e18),
            (-9223372036854775808, -9.223372e18),
        ];

        for i in 0..inputs.len() {
            let (x, f) = inputs[i];
            let a = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
            let r = _mm_cvtsi64_ss(a, x);
            let e = _mm_setr_ps(f, 6.0, 7.0, 8.0);
            assert_eq_m128(e, r);
        }
    }
    #[cfg(target_arch = "x86_64")]
    test_mm_cvtsi64_ss();

    #[target_feature(enable = "sse")]
    unsafe fn test_mm_movemask_ps() {
        let r = _mm_movemask_ps(_mm_setr_ps(-1.0, 5.0, -5.0, 0.0));
        assert_eq!(r, 0b0101);

        let r = _mm_movemask_ps(_mm_setr_ps(-1.0, -5.0, -5.0, 0.0));
        assert_eq!(r, 0b0111);
    }
    test_mm_movemask_ps();

    let x = 0i8;
    _mm_prefetch(&x, _MM_HINT_T0);
    _mm_prefetch(&x, _MM_HINT_T1);
    _mm_prefetch(&x, _MM_HINT_T2);
    _mm_prefetch(&x, _MM_HINT_NTA);
    _mm_prefetch(&x, _MM_HINT_ET0);
    _mm_prefetch(&x, _MM_HINT_ET1);
}
