use super::copysignf;
use super::truncf;
use core::f32;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn roundf(x: f32) -> f32 {
    truncf(x + copysignf(0.5 - 0.25 * f32::EPSILON, x))
}

#[cfg(test)]
mod tests {
    use super::roundf;

    #[test]
    fn negative_zero() {
        assert_eq!(roundf(-0.0_f32).to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn sanity_check() {
        assert_eq!(roundf(-1.0), -1.0);
        assert_eq!(roundf(2.8), 3.0);
        assert_eq!(roundf(-0.5), -1.0);
        assert_eq!(roundf(0.5), 1.0);
        assert_eq!(roundf(-1.5), -2.0);
        assert_eq!(roundf(1.5), 2.0);
    }
}
