//! Masks that take up full SIMD vector registers.

/// The error type returned when converting an integer to a mask fails.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TryFromMaskError(());

impl core::fmt::Display for TryFromMaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(
            f,
            "mask vector must have all bits set or unset in each lane"
        )
    }
}

macro_rules! define_mask {
    {
        $(#[$attr:meta])*
        struct $name:ident<const $lanes:ident: usize>(
            crate::$type:ident<$lanes2:ident>
        );
    } => {
        $(#[$attr])*
        #[derive(Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name<const $lanes: usize>(crate::$type<$lanes2>)
        where
            crate::$type<LANES>: crate::LanesAtMost32;

        impl<const LANES: usize> Copy for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {}

        impl<const LANES: usize> Clone for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const LANES: usize> $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            /// Construct a mask by setting all lanes to the given value.
            pub fn splat(value: bool) -> Self {
                Self(<crate::$type<LANES>>::splat(
                    if value {
                        -1
                    } else {
                        0
                    }
                ))
            }

            /// Tests the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn test(&self, lane: usize) -> bool {
                assert!(lane < LANES, "lane index out of range");
                self.0[lane] == -1
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn set(&mut self, lane: usize, value: bool) {
                assert!(lane < LANES, "lane index out of range");
                self.0[lane] = if value {
                    -1
                } else {
                    0
                }
            }

            /// Converts the mask to the equivalent integer representation, where -1 represents
            /// "set" and 0 represents "unset".
            #[inline]
            pub fn to_int(self) -> crate::$type<LANES> {
                self.0
            }

            /// Creates a  mask from the equivalent integer representation, where -1 represents
            /// "set" and 0 represents "unset".
            ///
            /// Each provided lane must be either 0 or -1.
            #[inline]
            pub unsafe fn from_int_unchecked(value: crate::$type<LANES>) -> Self {
                Self(value)
            }

            /// Creates a mask from the equivalent integer representation, where -1 represents
            /// "set" and 0 represents "unset".
            ///
            /// # Panics
            /// Panics if any lane is not 0 or -1.
            #[inline]
            pub fn from_int(value: crate::$type<LANES>) -> Self {
                use core::convert::TryInto;
                value.try_into().unwrap()
            }
        }

        impl<const LANES: usize> core::convert::From<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn from(value: bool) -> Self {
                Self::splat(value)
            }
        }

        impl<const LANES: usize> core::convert::TryFrom<crate::$type<LANES>> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Error = TryFromMaskError;
            fn try_from(value: crate::$type<LANES>) -> Result<Self, Self::Error> {
                let valid = (value.lanes_eq(crate::$type::<LANES>::splat(0)) | value.lanes_eq(crate::$type::<LANES>::splat(-1))).all();
                if valid {
                    Ok(Self(value))
                } else {
                    Err(TryFromMaskError(()))
                }
            }
        }

        impl<const LANES: usize> core::convert::From<$name<LANES>> for crate::$type<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn from(value: $name<LANES>) -> Self {
                value.0
            }
        }

        impl<const LANES: usize> core::convert::From<crate::BitMask<LANES>> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
            crate::BitMask<LANES>: crate::LanesAtMost32,
        {
            fn from(value: crate::BitMask<LANES>) -> Self {
                // TODO use an intrinsic to do this efficiently (with LLVM's sext instruction)
                let mut mask = Self::splat(false);
                for lane in 0..LANES {
                    mask.set(lane, value.test(lane));
                }
                mask
            }
        }

        impl<const LANES: usize> core::convert::From<$name<LANES>> for crate::BitMask<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
            crate::BitMask<LANES>: crate::LanesAtMost32,
        {
            fn from(value: $name<$lanes>) -> Self {
                // TODO use an intrinsic to do this efficiently (with LLVM's trunc instruction)
                let mut mask = Self::splat(false);
                for lane in 0..LANES {
                    mask.set(lane, value.test(lane));
                }
                mask
            }
        }

        impl<const LANES: usize> core::fmt::Debug for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                f.debug_list()
                    .entries((0..LANES).map(|lane| self.test(lane)))
                    .finish()
            }
        }

        impl<const LANES: usize> core::fmt::Binary for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Binary::fmt(&self.0, f)
            }
        }

        impl<const LANES: usize> core::fmt::Octal for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Octal::fmt(&self.0, f)
            }
        }

        impl<const LANES: usize> core::fmt::LowerHex for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::LowerHex::fmt(&self.0, f)
            }
        }

        impl<const LANES: usize> core::fmt::UpperHex for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::UpperHex::fmt(&self.0, f)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: bool) -> Self {
                self & Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<$name<LANES>> for bool
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitand(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) & rhs
            }
        }

        impl<const LANES: usize> core::ops::BitOr for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: bool) -> Self {
                self | Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<$name<LANES>> for bool
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitor(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) | rhs
            }
        }

        impl<const LANES: usize> core::ops::BitXor for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: bool) -> Self::Output {
                self ^ Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<$name<LANES>> for bool
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitxor(self, rhs: $name<LANES>) -> Self::Output {
                $name::<LANES>::splat(self) ^ rhs
            }
        }

        impl<const LANES: usize> core::ops::Not for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            type Output = $name<LANES>;
            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn bitand_assign(&mut self, rhs: bool) {
                *self &= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn bitor_assign(&mut self, rhs: bool) {
                *self |= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign<bool> for $name<LANES>
        where
            crate::$type<LANES>: crate::LanesAtMost32,
        {
            #[inline]
            fn bitxor_assign(&mut self, rhs: bool) {
                *self ^= Self::splat(rhs);
            }
        }

        impl_full_mask_reductions! { $name, $type }
    }
}

define_mask! {
    /// A mask equivalent to [SimdI8](crate::SimdI8), where all bits in the lane must be either set
    /// or unset.
    struct SimdMask8<const LANES: usize>(crate::SimdI8<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI16](crate::SimdI16), where all bits in the lane must be either set
    /// or unset.
    struct SimdMask16<const LANES: usize>(crate::SimdI16<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI32](crate::SimdI32), where all bits in the lane must be either set
    /// or unset.
    struct SimdMask32<const LANES: usize>(crate::SimdI32<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI64](crate::SimdI64), where all bits in the lane must be either set
    /// or unset.
    struct SimdMask64<const LANES: usize>(crate::SimdI64<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI128](crate::SimdI128), where all bits in the lane must be either set
    /// or unset.
    struct SimdMask128<const LANES: usize>(crate::SimdI128<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdIsize](crate::SimdIsize), where all bits in the lane must be either set
    /// or unset.
    struct SimdMaskSize<const LANES: usize>(crate::SimdIsize<LANES>);
}
