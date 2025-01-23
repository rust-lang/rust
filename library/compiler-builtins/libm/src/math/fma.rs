/// Fused multiply add (f64)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    return super::generic::fma(x, y, z);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fma_segfault() {
        // These two inputs cause fma to segfault on release due to overflow:
        assert_eq!(
            fma(
                -0.0000000000000002220446049250313,
                -0.0000000000000002220446049250313,
                -0.0000000000000002220446049250313
            ),
            -0.00000000000000022204460492503126,
        );

        let result = fma(-0.992, -0.992, -0.992);
        //force rounding to storage format on x87 to prevent superious errors.
        #[cfg(all(target_arch = "x86", not(target_feature = "sse2")))]
        let result = force_eval!(result);
        assert_eq!(result, -0.007936000000000007,);
    }

    #[test]
    fn fma_sbb() {
        assert_eq!(fma(-(1.0 - f64::EPSILON), f64::MIN, f64::MIN), -3991680619069439e277);
    }

    #[test]
    fn fma_underflow() {
        assert_eq!(fma(1.1102230246251565e-16, -9.812526705433188e-305, 1.0894e-320), 0.0,);
    }
}
