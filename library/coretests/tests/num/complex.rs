use core::num::Complex;

use crate::num::float_test;

float_test! {
    name: addition,
    attrs: {
        const: #[cfg(false)], // requires const traits
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test {
        type CFloat = Complex<Float>;
        assert_eq!(CFloat::new(1.0, 2.0) + CFloat::new(3.0, 4.0), CFloat::new(4.0, 6.0));
        assert_eq!(CFloat::new(0.0, 2.0) + CFloat::new(0.0, 4.0), CFloat::new(0.0, 6.0));
        assert_eq!(CFloat::new(1.0, 0.0) + CFloat::new(3.0, 0.0), CFloat::new(4.0, 0.0));
        assert_eq!(CFloat::new(1.0, 0.0) + 1.0, CFloat::new(2.0, 0.0));
    }
}
float_test! {
    name: subtraction,
    attrs: {
        const: #[cfg(false)], // requires const traits
        f16: #[cfg(any(miri, target_has_reliable_f16))],
        f128: #[cfg(any(miri, target_has_reliable_f128))],
    },
    test {
        type CFloat = Complex<Float>;
        assert_eq!(CFloat::new(3.0, 4.0) - CFloat::new(1.0, 2.0), CFloat::new(2.0, 2.0));
        assert_eq!(CFloat::new(3.0, 4.0) - CFloat::new(0.0, 2.0), CFloat::new(3.0, 2.0));
        assert_eq!(CFloat::new(3.0, 4.0) - CFloat::new(1.0, 0.0), CFloat::new(2.0, 4.0));
        assert_eq!(CFloat::new(1.0, 0.0) - 1.0, CFloat::new(0.0, 0.0));
    }
}
