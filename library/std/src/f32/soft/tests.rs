use crate::f32::soft::{Positive, PositiveFinite, Representation, Sign, pow2};

#[test]
fn test_representation() {
    assert_eq!(Representation::new(-1.5f32), Representation {
        sign: Sign::Negative,
        abs: Positive::Finite(PositiveFinite { exp: -23, mantissa: 0xc00000 }),
    });

    assert_eq!(Representation::new(f32::MIN_POSITIVE), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -149, mantissa: 0x800000 }),
    });

    assert_eq!(Representation::new(f32::MIN_POSITIVE / 2.0), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -150, mantissa: 0x800000 }),
    });

    assert_eq!(Representation::new(f32::MAX), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: 104, mantissa: 0xffffff }),
    });

    assert_eq!(Representation::new(-0.0), Representation {
        sign: Sign::Negative,
        abs: Positive::Zero,
    });

    assert_eq!(Representation::new(f32::INFINITY), Representation {
        sign: Sign::Positive,
        abs: Positive::Infinity,
    });

    assert_eq!(Representation::new(f32::NAN), Representation {
        sign: Sign::Positive,
        abs: Positive::NaN,
    });
}

#[test]
fn test_pow2() {
    assert_eq!(pow2(2), 4.0);
    assert_eq!(pow2(-1), 0.5);
    assert_eq!(pow2(-150), 0.0);
    assert_eq!(pow2(-126), f32::MIN_POSITIVE);
    assert_eq!(pow2(-127), f32::MIN_POSITIVE / 2.0);
    assert_eq!(pow2(128), f32::INFINITY);
}
