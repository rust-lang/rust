//! Compare numeric types approximately.

use float_cmp::Ulps;

pub trait ApproxEq {
    fn approxeq(&self, other: &Self, _ulps: i64) -> bool;
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result;
}

impl ApproxEq for bool {
    fn approxeq(&self, other: &Self, _ulps: i64) -> bool {
        self == other
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

macro_rules! impl_integer_approxeq {
    { $($type:ty),* } => {
        $(
        impl ApproxEq for $type {
            fn approxeq(&self, other: &Self, _ulps: i64) -> bool {
                self == other
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self)
            }
        }
        )*
    };
}

impl_integer_approxeq! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize }

macro_rules! impl_float_approxeq {
    { $($type:ty),* } => {
        $(
        impl ApproxEq for $type {
            fn approxeq(&self, other: &Self, ulps: i64) -> bool {
                if self.is_nan() && other.is_nan() {
                    true
                } else {
                    (self.ulps(other) as i64).abs() <= ulps
                }
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self.to_bits())
            }
        }
        )*
    };
}

impl_float_approxeq! { f32, f64 }

impl<T: ApproxEq, const N: usize> ApproxEq for [T; N] {
    fn approxeq(&self, other: &Self, ulps: i64) -> bool {
        self.iter()
            .zip(other.iter())
            .fold(true, |value, (left, right)| {
                value && left.approxeq(right, ulps)
            })
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[repr(transparent)]
        struct Wrapper<'a, T: ApproxEq>(&'a T);

        impl<T: ApproxEq> core::fmt::Debug for Wrapper<'_, T> {
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
pub struct ApproxEqWrapper<'a, T>(pub &'a T, pub i64);

impl<T: ApproxEq> PartialEq<T> for ApproxEqWrapper<'_, T> {
    fn eq(&self, other: &T) -> bool {
        self.0.approxeq(other, self.1)
    }
}

impl<T: ApproxEq> core::fmt::Debug for ApproxEqWrapper<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

#[macro_export]
macro_rules! prop_assert_approxeq {
    { $a:expr, $b:expr, $ulps:expr $(,)? } => {
        {
            use $crate::approxeq::ApproxEqWrapper;
            let a = $a;
            let b = $b;
            proptest::prop_assert_eq!(ApproxEqWrapper(&a, $ulps), b);
        }
    };
}
