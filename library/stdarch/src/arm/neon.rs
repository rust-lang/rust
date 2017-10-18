//! ARM NEON intrinsics
//!
//! The references is [ARM's NEON Intrinsics Reference](http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf). [ARM's NEON Intrinsics Online Database](https://developer.arm.com/technologies/neon/intrinsics) is also useful.

#[cfg(test)]
use stdsimd_test::assert_instr;

use v64::{f32x2};

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.aarch64.neon.frsqrte.v2f32"]
    fn frsqrte_v2f32(a: f32x2) -> f32x2;
}

/// Reciprocal square-root estimate.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(frsqrte))]
pub unsafe fn vrsqrte_f32(a: f32x2) -> f32x2 {
    frsqrte_v2f32(a)
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v64::{f32x2};
    use arm::neon;

    #[test]
    fn vrsqrt_f32() {
        let a = f32x2::new(1.0, 2.0);
        let e = f32x2::new(0.9980469, 0.7050781);
        let r = unsafe { neon::vrsqrte_f32(a) };
        assert_eq!(r, e);
    }
}
