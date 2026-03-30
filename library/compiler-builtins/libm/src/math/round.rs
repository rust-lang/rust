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
    use crate::support::{Float, Hex};

    macro_rules! cases {
        ($f:ty) => {
            [
                // roundtrip
                (0.0, 0.0),
                (-0.0, -0.0),
                (1.0, 1.0),
                (-1.0, -1.0),
                (<$f>::INFINITY, <$f>::INFINITY),
                (<$f>::NEG_INFINITY, <$f>::NEG_INFINITY),
                // with rounding
                (0.1, 0.0),
                (-0.1, -0.0),
                (0.5, 1.0),
                (-0.5, -1.0),
                (0.9, 1.0),
                (-0.9, -1.0),
                (1.1, 1.0),
                (-1.1, -1.0),
                (1.5, 2.0),
                (-1.5, -2.0),
                (1.9, 2.0),
                (-1.9, -2.0),
            ]
        };
    }

    #[track_caller]
    fn check<F: Float>(f: fn(F) -> F, cases: &[(F, F)]) {
        for &(x, exp_res) in cases {
            let val = generic::round(x);
            assert_biteq!(val, exp_res, "generic::round_status({x:?}) {}", Hex(x));
            let val = f(x);
            assert_biteq!(val, exp_res, "round({x:?}) {}", Hex(x));
        }
    }

    #[test]
    #[cfg(f16_enabled)]
    fn check_f16() {
        check::<f16>(roundf16, &cases!(f16));
        check::<f16>(
            roundf16,
            &[
                (hf16!("0x1p10"), hf16!("0x1p10")),
                (hf16!("-0x1p10"), hf16!("-0x1p10")),
            ],
        );
    }

    #[test]
    fn check_f32() {
        check::<f32>(roundf, &cases!(f32));
        check::<f32>(
            roundf,
            &[
                (hf32!("0x1p23"), hf32!("0x1p23")),
                (hf32!("-0x1p23"), hf32!("-0x1p23")),
            ],
        );
    }

    #[test]
    fn check_f64() {
        check::<f64>(round, &cases!(f64));
        check::<f64>(
            round,
            &[
                (hf64!("0x1p52"), hf64!("0x1p52")),
                (hf64!("-0x1p52"), hf64!("-0x1p52")),
            ],
        );
    }

    #[test]
    #[cfg(f128_enabled)]
    fn check_f128() {
        check::<f128>(roundf128, &cases!(f128));
        check::<f128>(
            roundf128,
            &[
                (hf128!("0x1p112"), hf128!("0x1p112")),
                (hf128!("-0x1p112"), hf128!("-0x1p112")),
            ],
        );
    }
}
