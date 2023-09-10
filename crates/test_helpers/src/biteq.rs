//! Compare numeric types by exact bit value.

pub trait BitEq {
    fn biteq(&self, other: &Self) -> bool;
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result;
}

impl BitEq for bool {
    fn biteq(&self, other: &Self) -> bool {
        self == other
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

macro_rules! impl_integer_biteq {
    { $($type:ty),* } => {
        $(
        impl BitEq for $type {
            fn biteq(&self, other: &Self) -> bool {
                self == other
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self)
            }
        }
        )*
    };
}

impl_integer_biteq! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize }

macro_rules! impl_float_biteq {
    { $($type:ty),* } => {
        $(
        impl BitEq for $type {
            fn biteq(&self, other: &Self) -> bool {
                if self.is_nan() && other.is_nan() {
                    true // exact nan bits don't matter
                } else {
                    self.to_bits() == other.to_bits()
                }
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self.to_bits())
            }
        }
        )*
    };
}

impl_float_biteq! { f32, f64 }

impl<T> BitEq for *const T {
    fn biteq(&self, other: &Self) -> bool {
        self == other
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T> BitEq for *mut T {
    fn biteq(&self, other: &Self) -> bool {
        self == other
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<T: BitEq, const N: usize> BitEq for [T; N] {
    fn biteq(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .fold(true, |value, (left, right)| value && left.biteq(right))
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[repr(transparent)]
        struct Wrapper<'a, T: BitEq>(&'a T);

        impl<T: BitEq> core::fmt::Debug for Wrapper<'_, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                self.0.fmt(f)
            }
        }

        f.debug_list()
            .entries(self.iter().map(|x| Wrapper(x)))
            .finish()
    }
}

#[doc(hidden)]
pub struct BitEqWrapper<'a, T>(pub &'a T);

impl<T: BitEq> PartialEq for BitEqWrapper<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.biteq(other.0)
    }
}

impl<T: BitEq> core::fmt::Debug for BitEqWrapper<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

#[doc(hidden)]
pub struct BitEqEitherWrapper<'a, T>(pub &'a T, pub &'a T);

impl<T: BitEq> PartialEq<BitEqEitherWrapper<'_, T>> for BitEqWrapper<'_, T> {
    fn eq(&self, other: &BitEqEitherWrapper<'_, T>) -> bool {
        self.0.biteq(other.0) || self.0.biteq(other.1)
    }
}

impl<T: BitEq> core::fmt::Debug for BitEqEitherWrapper<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.0.biteq(self.1) {
            self.0.fmt(f)
        } else {
            self.0.fmt(f)?;
            write!(f, " or ")?;
            self.1.fmt(f)
        }
    }
}

#[macro_export]
macro_rules! prop_assert_biteq {
    { $a:expr, $b:expr $(,)? } => {
        {
            use $crate::biteq::BitEqWrapper;
            let a = $a;
            let b = $b;
            proptest::prop_assert_eq!(BitEqWrapper(&a), BitEqWrapper(&b));
        }
    };
    { $a:expr, $b:expr, $c:expr $(,)? } => {
        {
            use $crate::biteq::{BitEqWrapper, BitEqEitherWrapper};
            let a = $a;
            let b = $b;
            let c = $c;
            proptest::prop_assert_eq!(BitEqWrapper(&a), BitEqEitherWrapper(&b, &c));
        }
    };
}
