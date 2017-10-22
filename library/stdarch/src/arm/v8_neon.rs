//! ARMv8 NEON intrinsics

#[cfg(test)]
use stdsimd_test::assert_instr;
use simd_llvm::simd_add;
use v128::f64x2;

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vadd_f64(a: f64, b: f64) -> f64 {
    a + b
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(fadd))]
pub unsafe fn vaddq_f64(a: f64x2, b: f64x2) -> f64x2 {
    simd_add(a, b)
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_s64(a: i64, b: i64) -> i64 {
    a + b
}

/// Vector add.
#[inline(always)]
#[target_feature = "+neon"]
#[cfg_attr(test, assert_instr(add))]
pub unsafe fn vaddd_u64(a: u64, b: u64) -> u64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vadd_f64_() {
        let a = 1.;
        let b = 8.;
        let e = 9.;
        let r = unsafe { vadd_f64(a, b) };
        assert_eq!(r, e);
    }

    #[test]
    fn vaddq_f64_() {
        let a = f64x2::new(1., 2.);
        let b = f64x2::new(8., 7.);
        let e = f64x2::new(9., 9.);
        let r = unsafe { vaddq_f64(a, b) };
        assert_eq!(r, e);
    }

    #[test]
    fn vaddd_s64_() {
        let a = 1;
        let b = 8;
        let e = 9;
        let r = unsafe { vaddd_s64(a, b) };
        assert_eq!(r, e);
    }

    #[test]
    fn vaddd_u64_() {
        let a = 1;
        let b = 8;
        let e = 9;
        let r = unsafe { vaddd_u64(a, b) };
        assert_eq!(r, e);
    }
}
