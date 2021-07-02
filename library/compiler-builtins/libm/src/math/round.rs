use super::copysign;
use super::trunc;
use core::f64;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn round(x: f64) -> f64 {
    trunc(x + copysign(0.5 - 0.25 * f64::EPSILON, x))
}

#[cfg(test)]
mod tests {
    use super::round;

    #[test]
    fn negative_zero() {
        assert_eq!(round(-0.0_f64).to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn sanity_check() {
        assert_eq!(round(-1.0), -1.0);
        assert_eq!(round(2.8), 3.0);
        assert_eq!(round(-0.5), -1.0);
        assert_eq!(round(0.5), 1.0);
        assert_eq!(round(-1.5), -2.0);
        assert_eq!(round(1.5), 2.0);
    }
}
