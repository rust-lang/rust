use crate::simd::intrinsics::{simd_saturating_add, simd_saturating_sub};
use crate::simd::{LaneCount, Simd, SupportedLaneCount};

macro_rules! impl_uint_arith {
    ($($ty:ty),+) => {
        $( impl<const LANES: usize> Simd<$ty, LANES> where LaneCount<LANES>: SupportedLaneCount {

            /// Lanewise saturating add.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::MAX;")]
            /// let x = Simd::from_array([2, 1, 0, MAX]);
            /// let max = Simd::splat(MAX);
            /// let unsat = x + max;
            /// let sat = x.saturating_add(max);
            /// assert_eq!(x - 1, unsat);
            /// assert_eq!(sat, max);
            /// ```
            #[inline]
            pub fn saturating_add(self, second: Self) -> Self {
                unsafe { simd_saturating_add(self, second) }
            }

            /// Lanewise saturating subtract.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::MAX;")]
            /// let x = Simd::from_array([2, 1, 0, MAX]);
            /// let max = Simd::splat(MAX);
            /// let unsat = x - max;
            /// let sat = x.saturating_sub(max);
            /// assert_eq!(unsat, x + 1);
            /// assert_eq!(sat, Simd::splat(0));
            #[inline]
            pub fn saturating_sub(self, second: Self) -> Self {
                unsafe { simd_saturating_sub(self, second) }
            }
        })+
    }
}

macro_rules! impl_int_arith {
    ($($ty:ty),+) => {
        $( impl<const LANES: usize> Simd<$ty, LANES> where LaneCount<LANES>: SupportedLaneCount {

            /// Lanewise saturating add.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::{MIN, MAX};")]
            /// let x = Simd::from_array([MIN, 0, 1, MAX]);
            /// let max = Simd::splat(MAX);
            /// let unsat = x + max;
            /// let sat = x.saturating_add(max);
            /// assert_eq!(unsat, Simd::from_array([-1, MAX, MIN, -2]));
            /// assert_eq!(sat, Simd::from_array([-1, MAX, MAX, MAX]));
            /// ```
            #[inline]
            pub fn saturating_add(self, second: Self) -> Self {
                unsafe { simd_saturating_add(self, second) }
            }

            /// Lanewise saturating subtract.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::{MIN, MAX};")]
            /// let x = Simd::from_array([MIN, -2, -1, MAX]);
            /// let max = Simd::splat(MAX);
            /// let unsat = x - max;
            /// let sat = x.saturating_sub(max);
            /// assert_eq!(unsat, Simd::from_array([1, MAX, MIN, 0]));
            /// assert_eq!(sat, Simd::from_array([MIN, MIN, MIN, 0]));
            #[inline]
            pub fn saturating_sub(self, second: Self) -> Self {
                unsafe { simd_saturating_sub(self, second) }
            }

            /// Lanewise absolute value, implemented in Rust.
            /// Every lane becomes its absolute value.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::{MIN, MAX};")]
            /// let xs = Simd::from_array([MIN, MIN +1, -5, 0]);
            /// assert_eq!(xs.abs(), Simd::from_array([MIN, MAX, 5, 0]));
            /// ```
            #[inline]
            pub fn abs(self) -> Self {
                const SHR: $ty = <$ty>::BITS as $ty - 1;
                let m = self >> SHR;
                (self^m) - m
            }

            /// Lanewise saturating absolute value, implemented in Rust.
            /// As abs(), except the MIN value becomes MAX instead of itself.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::{MIN, MAX};")]
            /// let xs = Simd::from_array([MIN, -2, 0, 3]);
            /// let unsat = xs.abs();
            /// let sat = xs.saturating_abs();
            /// assert_eq!(unsat, Simd::from_array([MIN, 2, 0, 3]));
            /// assert_eq!(sat, Simd::from_array([MAX, 2, 0, 3]));
            /// ```
            #[inline]
            pub fn saturating_abs(self) -> Self {
                // arith shift for -1 or 0 mask based on sign bit, giving 2s complement
                const SHR: $ty = <$ty>::BITS as $ty - 1;
                let m = self >> SHR;
                (self^m).saturating_sub(m)
            }

            /// Lanewise saturating negation, implemented in Rust.
            /// As neg(), except the MIN value becomes MAX instead of itself.
            ///
            /// # Examples
            /// ```
            /// # #![feature(portable_simd)]
            /// # #[cfg(feature = "std")] use core_simd::Simd;
            /// # #[cfg(not(feature = "std"))] use core::simd::Simd;
            #[doc = concat!("# use core::", stringify!($ty), "::{MIN, MAX};")]
            /// let x = Simd::from_array([MIN, -2, 3, MAX]);
            /// let unsat = -x;
            /// let sat = x.saturating_neg();
            /// assert_eq!(unsat, Simd::from_array([MIN, 2, -3, MIN + 1]));
            /// assert_eq!(sat, Simd::from_array([MAX, 2, -3, MIN + 1]));
            /// ```
            #[inline]
            pub fn saturating_neg(self) -> Self {
                Self::splat(0).saturating_sub(self)
            }
        })+
    }
}

impl_uint_arith! { u8, u16, u32, u64, usize }
impl_int_arith! { i8, i16, i32, i64, isize }
