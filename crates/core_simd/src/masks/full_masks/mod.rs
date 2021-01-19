//! Masks that take up full SIMD vector registers.

/// The error type returned when converting an integer to a mask fails.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TryFromMaskError(());

impl core::fmt::Display for TryFromMaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "mask vector must have all bits set or unset in each lane")
    }
}

macro_rules! define_mask {
    { $(#[$attr:meta])* struct $name:ident<const $lanes:ident: usize>($type:ty); } => {
        $(#[$attr])*
        #[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name<const $lanes: usize>($type);

        delegate_ops_to_inner! { $name }

        impl<const $lanes: usize> $name<$lanes> {
            /// Construct a mask by setting all lanes to the given value.
            pub fn splat(value: bool) -> Self {
                Self(<$type>::splat(
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
                self.0[lane] == -1
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn set(&mut self, lane: usize, value: bool) {
                self.0[lane] = if value {
                    -1
                } else {
                    0
                }
            }
        }

        impl<const $lanes: usize> core::convert::From<bool> for $name<$lanes> {
            fn from(value: bool) -> Self {
                Self::splat(value)
            }
        }

        impl<const $lanes: usize> core::convert::TryFrom<$type> for $name<$lanes> {
            type Error = TryFromMaskError;
            fn try_from(value: $type) -> Result<Self, Self::Error> {
                if value.as_slice().iter().all(|x| *x == 0 || *x == -1) {
                    Ok(Self(value))
                } else {
                    Err(TryFromMaskError(()))
                }
            }
        }

        impl<const $lanes: usize> core::convert::From<$name<$lanes>> for $type {
            fn from(value: $name<$lanes>) -> Self {
                value.0
            }
        }

        impl<const $lanes: usize> core::fmt::Debug for $name<$lanes> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                f.debug_list()
                    .entries((0..LANES).map(|lane| self.test(lane)))
                    .finish()
            }
        }

        impl<const $lanes: usize> core::fmt::Binary for $name<$lanes> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Binary::fmt(&self.0, f)
            }
        }

        impl<const $lanes: usize> core::fmt::Octal for $name<$lanes> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Octal::fmt(&self.0, f)
            }
        }

        impl<const $lanes: usize> core::fmt::LowerHex for $name<$lanes> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::LowerHex::fmt(&self.0, f)
            }
        }

        impl<const $lanes: usize> core::fmt::UpperHex for $name<$lanes> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::UpperHex::fmt(&self.0, f)
            }
        }
    }
}

define_mask! {
    /// A mask equivalent to [SimdI8](crate::SimdI8), where all bits in the lane must be either set
    /// or unset.
    struct SimdI8Mask<const LANES: usize>(crate::SimdI8<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI16](crate::SimdI16), where all bits in the lane must be either set
    /// or unset.
    struct SimdI16Mask<const LANES: usize>(crate::SimdI16<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI32](crate::SimdI32), where all bits in the lane must be either set
    /// or unset.
    struct SimdI32Mask<const LANES: usize>(crate::SimdI32<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI64](crate::SimdI64), where all bits in the lane must be either set
    /// or unset.
    struct SimdI64Mask<const LANES: usize>(crate::SimdI64<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdI128](crate::SimdI128), where all bits in the lane must be either set
    /// or unset.
    struct SimdI128Mask<const LANES: usize>(crate::SimdI64<LANES>);
}

define_mask! {
    /// A mask equivalent to [SimdIsize](crate::SimdIsize), where all bits in the lane must be either set
    /// or unset.
    struct SimdIsizeMask<const LANES: usize>(crate::SimdI64<LANES>);
}

macro_rules! implement_mask_ext {
    { $($vector:ident => $mask:ident,)* } => {
        $(
            impl<const LANES: usize> crate::masks::MaskExt<$mask<LANES>> for crate::$vector<LANES> {
                #[inline]
                fn lanes_eq(&self, other: &Self) -> $mask<LANES> {
                    unsafe { crate::intrinsics::simd_eq(self, other) }
                }

                #[inline]
                fn lanes_ne(&self, other: &Self) -> $mask<LANES> {
                    unsafe { crate::intrinsics::simd_ne(self, other) }
                }

                #[inline]
                fn lanes_lt(&self, other: &Self) -> $mask<LANES> {
                    unsafe { crate::intrinsics::simd_lt(self, other) }
                }

                #[inline]
                fn lanes_gt(&self, other: &Self) -> $mask<LANES> {
                    unsafe { crate::intrinsics::simd_gt(self, other) }
                }

                #[inline]
                fn lanes_le(&self, other: &Self) -> $mask<LANES> {
                    unsafe { crate::intrinsics::simd_le(self, other) }
                }

                #[inline]
                fn lanes_ge(&self, other: &Self) -> $mask<LANES> {
                    unsafe { crate::intrinsics::simd_ge(self, other) }
                }
            }
        )*
    }
}

implement_mask_ext! {
    SimdI8 => SimdI8Mask,
    SimdI16 => SimdI16Mask,
    SimdI32 => SimdI32Mask,
    SimdI64 => SimdI64Mask,
    SimdI128 => SimdI128Mask,
    SimdIsize => SimdIsizeMask,

    SimdU8 => SimdI8Mask,
    SimdU16 => SimdI16Mask,
    SimdU32 => SimdI32Mask,
    SimdU64 => SimdI64Mask,
    SimdU128 => SimdI128Mask,
    SimdUsize => SimdIsizeMask,

    SimdF32 => SimdI32Mask,
    SimdF64 => SimdI64Mask,
}
