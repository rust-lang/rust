macro_rules! impl_uint_arith {
    ($(($name:ident, $n:ty)),+) => {
        $( impl<const LANES: usize> $name<LANES> where Self: crate::LanesAtMost32 {

            /// Lanewise saturating add.
            ///
            /// # Examples
            /// ```
            /// # use core_simd::*;
            #[doc = concat!("# use core::", stringify!($n), "::MAX;")]
            #[doc = concat!("let x = ", stringify!($name), "::from_array([2, 1, 0, MAX]);")]
            #[doc = concat!("let max = ", stringify!($name), "::splat(MAX);")]
            /// let unsat = x + max;
            /// let sat = x.saturating_add(max);
            /// assert_eq!(x - 1, unsat);
            /// assert_eq!(sat, max);
            /// ```
            #[inline]
            pub fn saturating_add(self, second: Self) -> Self {
                unsafe { crate::intrinsics::simd_saturating_add(self, second) }
            }

            /// Lanewise saturating subtract.
            ///
            /// # Examples
            /// ```
            /// # use core_simd::*;
            #[doc = concat!("# use core::", stringify!($n), "::MAX;")]
            #[doc = concat!("let x = ", stringify!($name), "::from_array([2, 1, 0, MAX]);")]
            #[doc = concat!("let max = ", stringify!($name), "::splat(MAX);")]
            /// let unsat = x - max;
            /// let sat = x.saturating_sub(max);
            /// assert_eq!(unsat, x + 1);
            #[doc = concat!("assert_eq!(sat, ", stringify!($name), "::splat(0));")]
            #[inline]
            pub fn saturating_sub(self, second: Self) -> Self {
                unsafe { crate::intrinsics::simd_saturating_sub(self, second) }
            }
        })+
    }
}

macro_rules! impl_int_arith {
    ($(($name:ident, $n:ty)),+) => {
        $( impl<const LANES: usize> $name<LANES> where Self: crate::LanesAtMost32 {

            /// Lanewise saturating add.
            ///
            /// # Examples
            /// ```
            /// # use core_simd::*;
            #[doc = concat!("# use core::", stringify!($n), "::{MIN, MAX};")]
            #[doc = concat!("let x = ", stringify!($name), "::from_array([MIN, 0, 1, MAX]);")]
            #[doc = concat!("let max = ", stringify!($name), "::splat(MAX);")]
            /// let unsat = x + max;
            /// let sat = x.saturating_add(max);
            #[doc = concat!("assert_eq!(unsat, ", stringify!($name), "::from_array([-1, MAX, MIN, -2]));")]
            #[doc = concat!("assert_eq!(sat, ", stringify!($name), "::from_array([-1, MAX, MAX, MAX]));")]
            /// ```
            #[inline]
            pub fn saturating_add(self, second: Self) -> Self {
                unsafe { crate::intrinsics::simd_saturating_add(self, second) }
            }

            /// Lanewise saturating subtract.
            ///
            /// # Examples
            /// ```
            /// # use core_simd::*;
            #[doc = concat!("# use core::", stringify!($n), "::{MIN, MAX};")]
            #[doc = concat!("let x = ", stringify!($name), "::from_array([MIN, -2, -1, MAX]);")]
            #[doc = concat!("let max = ", stringify!($name), "::splat(MAX);")]
            /// let unsat = x - max;
            /// let sat = x.saturating_sub(max);
            #[doc = concat!("assert_eq!(unsat, ", stringify!($name), "::from_array([1, MAX, MIN, 0]));")]
            #[doc = concat!("assert_eq!(sat, ", stringify!($name), "::from_array([MIN, MIN, MIN, 0]));")]
            #[inline]
            pub fn saturating_sub(self, second: Self) -> Self {
                unsafe { crate::intrinsics::simd_saturating_sub(self, second) }
            }

            /// Lanewise saturating absolute value, implemented in Rust.
            /// As abs(), except the MIN value becomes MAX instead of itself.
            ///
            /// # Examples
            /// # use core_simd::*;
            #[doc = concat!("# use core::", stringify!($n), "::{MIN, MAX};")]
            #[doc = concat!("let x = ", stringify!($name), "::splat([MIN, -2, 0, 3]);")]
            /// let unsat = x.abs();
            /// let sat = x.saturating_abs();
            #[doc = concat!("assert_eq!(unsat, ", stringify!($name), "::from_array([MIN, 2, 0, 3]);")]
            #[doc = concat!("assert_eq!(sat, ", stringify!($name), "::from_array([MAX, 2, 0, 3]));")]
            /// ```
            #[inline]
            pub fn saturating_abs(self) -> Self {
                // arith shift for -1 or 0 mask based on sign bit, giving 2s complement
                const SHR: $n = <$n>::BITS as $n - 1;
                let m = self >> SHR;
                (self^m).saturating_sub(m)
            }

            /// Lanewise saturating negation, implemented in Rust.
            /// As neg(), except the MIN value becomes MAX instead of itself.
            ///
            /// # Examples
            /// # use core_simd::*;
            #[doc = concat!("# use core::", stringify!($n), "::{MIN, MAX};")]
            #[doc = concat!("let x = ", stringify!($name), "::splat([MIN, -2, 3, MAX]);")]
            /// let unsat = -x;
            /// let sat = x.saturating_neg();
            #[doc = concat!("assert_eq!(unsat, ", stringify!($name), "::from_array([MIN, 2, -3, MIN + 1]);")]
            #[doc = concat!("assert_eq!(sat, ", stringify!($name), "::from_array([MAX, 2, -3, MIN + 1]));")]
            /// ```
            #[inline]
            pub fn saturating_neg(self) -> Self {
                Self::splat(0).saturating_sub(self)
            }
        })+
    }
}

use crate::vector::*;

impl_uint_arith! { (SimdU8, u8), (SimdU16, u16), (SimdU32, u32), (SimdU64, u64), (SimdUsize, usize) }
impl_int_arith! { (SimdI8, i8), (SimdI16, i16), (SimdI32, i32), (SimdI64, i64), (SimdIsize, isize) }
