use crate::f64::soft::{Positive, PositiveFinite, Representation, Sign, pow2};

#[test]
fn test_representation() {
    assert_eq!(Representation::new(-1.5f64), Representation {
        sign: Sign::Negative,
        abs: Positive::Finite(PositiveFinite { exp: -52, mantissa: 3 << 51 }),
    });

    assert_eq!(Representation::new(f64::MIN_POSITIVE), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -1074, mantissa: 1 << 52 }),
    });

    assert_eq!(Representation::new(f64::MIN_POSITIVE / 2.0), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -1075, mantissa: 1 << 52 }),
    });

    assert_eq!(Representation::new(f64::MAX), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: 971, mantissa: (1 << 53) - 1 }),
    });

    assert_eq!(Representation::new(-0.0f64), Representation {
        sign: Sign::Negative,
        abs: Positive::Zero,
    });

    assert_eq!(Representation::new(f64::INFINITY), Representation {
        sign: Sign::Positive,
        abs: Positive::Infinity,
    });

    assert_eq!(Representation::new(f64::NAN), Representation {
        sign: Sign::Positive,
        abs: Positive::NaN,
    });
}

#[test]
fn test_pow2() {
    assert_eq!(pow2(2), 4.0);
    assert_eq!(pow2(-1), 0.5);
    assert_eq!(pow2(-1075), 0.0);
    assert_eq!(pow2(-1022), f64::MIN_POSITIVE);
    assert_eq!(pow2(-1023), f64::MIN_POSITIVE / 2.0);
    assert_eq!(pow2(1024), f64::INFINITY);
}
