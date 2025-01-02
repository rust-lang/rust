/// Absolute value (magnitude) (f64)
/// Calculates the absolute value (magnitude) of the argument `x`,
/// by direct manipulation of the bit representation of `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fabs(x: f64) -> f64 {
    select_implementation! {
        name: fabs,
        use_intrinsic: target_arch = "wasm32",
        args: x,
    }

    super::generic::fabs(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_check() {
        assert_eq!(fabs(-1.0), 1.0);
        assert_eq!(fabs(2.8), 2.8);
    }

    /// The spec: https://en.cppreference.com/w/cpp/numeric/math/fabs
    #[test]
    fn spec_tests() {
        assert!(fabs(f64::NAN).is_nan());
        for f in [0.0, -0.0].iter().copied() {
            assert_eq!(fabs(f), 0.0);
        }
        for f in [f64::INFINITY, f64::NEG_INFINITY].iter().copied() {
            assert_eq!(fabs(f), f64::INFINITY);
        }
    }
}
