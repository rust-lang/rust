// We're testing x86 target specific features
//@only-target: x86_64 i686
//@compile-flags: -C target-feature=+pclmulqdq

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    assert!(is_x86_feature_detected!("pclmulqdq"));

    let a = (0x7fffffffffffffff, 0x4317e40ab4ddcf05);
    let b = (0xdd358416f52ecd34, 0x633d11cc638ca16b);

    unsafe {
        assert_eq!(clmulepi64_si128::<0x00>(a, b), (13036940098130298092, 2704901987789626761));
        assert_eq!(clmulepi64_si128::<0x01>(a, b), (6707488474444649956, 3901733953304450635));
        assert_eq!(clmulepi64_si128::<0x10>(a, b), (11607166829323378905, 1191897396234301548));
        assert_eq!(clmulepi64_si128::<0x11>(a, b), (7731954893213347271, 1760130762532070957));
    }
}

#[target_feature(enable = "pclmulqdq")]
unsafe fn clmulepi64_si128<const IMM8: i32>(
    (a1, a2): (u64, u64),
    (b1, b2): (u64, u64),
) -> (u64, u64) {
    // SAFETY: There are no safety requirements for calling `_mm_clmulepi64_si128`.
    // It's just unsafe for API consistency with other intrinsics.
    unsafe {
        let a = core::mem::transmute::<_, __m128i>([a1, a2]);
        let b = core::mem::transmute::<_, __m128i>([b1, b2]);

        let out = _mm_clmulepi64_si128::<IMM8>(a, b);

        let [c1, c2] = core::mem::transmute::<_, [u64; 2]>(out);

        (c1, c2)
    }
}
