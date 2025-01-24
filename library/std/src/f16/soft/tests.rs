use crate::f16::soft::{Positive, PositiveFinite, Representation, Sign};

#[test]
fn test_representation() {
    assert_eq!(Representation::new(-1.5f16), Representation {
        sign: Sign::Negative,
        abs: Positive::Finite(PositiveFinite { exp: -10, mantissa: 0x600 }),
    });

    assert_eq!(Representation::new(f16::MIN_POSITIVE), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -24, mantissa: 0x400 }),
    });

    assert_eq!(Representation::new(f16::MIN_POSITIVE / 2.0), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: -25, mantissa: 0x400 }),
    });

    assert_eq!(Representation::new(f16::MAX), Representation {
        sign: Sign::Positive,
        abs: Positive::Finite(PositiveFinite { exp: 5, mantissa: 0x7ff }),
    });

    assert_eq!(Representation::new(-0.0), Representation {
        sign: Sign::Negative,
        abs: Positive::Zero,
    });

    assert_eq!(Representation::new(f16::INFINITY), Representation {
        sign: Sign::Positive,
        abs: Positive::Infinity,
    });

    assert_eq!(Representation::new(f16::NAN), Representation {
        sign: Sign::Positive,
        abs: Positive::NaN,
    });
}
