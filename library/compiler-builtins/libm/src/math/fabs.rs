use core::u64;

/// Absolute value (magnitude) (f64)
/// Calculates the absolute value (magnitude) of the argument `x`,
/// by direct manipulation of the bit representation of `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fabs(x: f64) -> f64 {
    // On wasm32 we know that LLVM's intrinsic will compile to an optimized
    // `f64.abs` native instruction, so we can leverage this for both code size
    // and speed.
    llvm_intrinsically_optimized! {
        #[cfg(target_arch = "wasm32")] {
            return unsafe { ::core::intrinsics::fabsf64(x) }
        }
    }
    f64::from_bits(x.to_bits() & (u64::MAX / 2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::*;

    #[test]
    fn sanity_check() {
        assert_eq!(fabs(-1.0), 1.0);
        assert_eq!(fabs(2.8), 2.8);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/fabs
    #[test]
    fn spec_tests() {
        assert!(fabs(NAN).is_nan());
        for f in [0.0, -0.0].iter().copied() {
            assert_eq!(fabs(f), 0.0);
        }
        for f in [INFINITY, NEG_INFINITY].iter().copied() {
            assert_eq!(fabs(f), INFINITY);
        }
    }
}
