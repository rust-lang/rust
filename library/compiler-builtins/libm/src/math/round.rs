use super::generic;

/// Round `x` to the nearest integer, breaking ties away from zero.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundf16(x: f16) -> f16 {
    generic::round(x)
}

/// Round `x` to the nearest integer, breaking ties away from zero.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundf(x: f32) -> f32 {
    generic::round(x)
}

/// Round `x` to the nearest integer, breaking ties away from zero.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn round(x: f64) -> f64 {
    generic::round(x)
}

/// Round `x` to the nearest integer, breaking ties away from zero.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn roundf128(x: f128) -> f128 {
    generic::round(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(f16_enabled)]
    fn zeroes_f16() {
        assert_biteq!(generic::round(0.0_f16), 0.0_f16);
        assert_biteq!(generic::round(-0.0_f16), -0.0_f16);
    }

    #[test]
    #[cfg(f16_enabled)]
    fn sanity_check_f16() {
        assert_eq!(generic::round(-1.0_f16), -1.0);
        assert_eq!(generic::round(2.8_f16), 3.0);
        assert_eq!(generic::round(-0.5_f16), -1.0);
        assert_eq!(generic::round(0.5_f16), 1.0);
        assert_eq!(generic::round(-1.5_f16), -2.0);
        assert_eq!(generic::round(1.5_f16), 2.0);
    }

    #[test]
    fn zeroes_f32() {
        assert_biteq!(generic::round(0.0_f32), 0.0_f32);
        assert_biteq!(generic::round(-0.0_f32), -0.0_f32);
    }

    #[test]
    fn sanity_check_f32() {
        assert_eq!(generic::round(-1.0_f32), -1.0);
        assert_eq!(generic::round(2.8_f32), 3.0);
        assert_eq!(generic::round(-0.5_f32), -1.0);
        assert_eq!(generic::round(0.5_f32), 1.0);
        assert_eq!(generic::round(-1.5_f32), -2.0);
        assert_eq!(generic::round(1.5_f32), 2.0);
    }

    #[test]
    fn zeroes_f64() {
        assert_biteq!(generic::round(0.0_f64), 0.0_f64);
        assert_biteq!(generic::round(-0.0_f64), -0.0_f64);
    }

    #[test]
    fn sanity_check_f64() {
        assert_eq!(generic::round(-1.0_f64), -1.0);
        assert_eq!(generic::round(2.8_f64), 3.0);
        assert_eq!(generic::round(-0.5_f64), -1.0);
        assert_eq!(generic::round(0.5_f64), 1.0);
        assert_eq!(generic::round(-1.5_f64), -2.0);
        assert_eq!(generic::round(1.5_f64), 2.0);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn zeroes_f128() {
        assert_biteq!(generic::round(0.0_f128), 0.0_f128);
        assert_biteq!(generic::round(-0.0_f128), -0.0_f128);
    }

    #[test]
    #[cfg(f128_enabled)]
    fn sanity_check_f128() {
        assert_eq!(generic::round(-1.0_f128), -1.0);
        assert_eq!(generic::round(2.8_f128), 3.0);
        assert_eq!(generic::round(-0.5_f128), -1.0);
        assert_eq!(generic::round(0.5_f128), 1.0);
        assert_eq!(generic::round(-1.5_f128), -2.0);
        assert_eq!(generic::round(1.5_f128), 2.0);
    }
}
