use crate::f128::soft::{Positive, PositiveFinite, Representation, Sign, pow2};

#[test]
fn test_representation() {
    assert_eq!(Representation::new(-1.5f128), Representation {
        sign: Sign::Negative,
        abs: Positive::Finite(PositiveFinite { exp: -112, mantissa: 3 << 111 }),
    });

    assert_eq!(Representation::new(f128::MIN_POSITIVE), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -16494, mantissa: 1 << 112 }),
    });

    assert_eq!(Representation::new(f128::MIN_POSITIVE / 2.0), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -16495, mantissa: 1 << 112 }),
    });

    assert_eq!(Representation::new(f128::MAX), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: 16271, mantissa: (1 << 113) - 1 }),
    });

    assert_eq!(Representation::new(-0.0f128), Representation {
        sign: Sign::Negative,
        abs: Positive::Zero,
    });

    assert_eq!(Representation::new(f128::INFINITY), Representation {
        sign: Sign::Positive,
        abs: Positive::Infinity,
    });

    assert_eq!(Representation::new(f128::NAN), Representation {
        sign: Sign::Positive,
        abs: Positive::NaN,
    });
}

#[test]
fn test_pow2() {
    assert_eq!(pow2(2), 4.0);
    assert_eq!(pow2(-1), 0.5);
    assert_eq!(pow2(-16495), 0.0);
    assert_eq!(pow2(-16382), f128::MIN_POSITIVE);
    assert_eq!(pow2(-16383), f128::MIN_POSITIVE / 2.0);
    assert_eq!(pow2(16384), f128::INFINITY);
}
