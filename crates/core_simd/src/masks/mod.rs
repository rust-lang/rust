//! Types and traits associated with masking lanes of vectors.
#![allow(non_camel_case_types)]

/// Implements bitwise ops on mask types by delegating the operators to the inner type.
macro_rules! delegate_ops_to_inner {
    { $name:ident } => {
        impl<const LANES: usize> core::ops::BitAnd for $name<LANES> {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<bool> for $name<LANES> {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: bool) -> Self {
                self & Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<$name<LANES>> for bool {
            type Output = $name<LANES>;
            #[inline]
            fn bitand(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) & rhs
            }
        }

        impl<const LANES: usize> core::ops::BitOr for $name<LANES> {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<bool> for $name<LANES> {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: bool) -> Self {
                self | Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<$name<LANES>> for bool {
            type Output = $name<LANES>;
            #[inline]
            fn bitor(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) | rhs
            }
        }

        impl<const LANES: usize> core::ops::BitXor for $name<LANES> {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<bool> for $name<LANES> {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: bool) -> Self::Output {
                self ^ Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<$name<LANES>> for bool {
            type Output = $name<LANES>;
            #[inline]
            fn bitxor(self, rhs: $name<LANES>) -> Self::Output {
                $name::<LANES>::splat(self) ^ rhs
            }
        }

        impl<const LANES: usize> core::ops::Not for $name<LANES> {
            type Output = $name<LANES>;
            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign for $name<LANES> {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign<bool> for $name<LANES> {
            #[inline]
            fn bitand_assign(&mut self, rhs: bool) {
                *self &= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign for $name<LANES> {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign<bool> for $name<LANES> {
            #[inline]
            fn bitor_assign(&mut self, rhs: bool) {
                *self |= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign for $name<LANES> {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign<bool> for $name<LANES> {
            #[inline]
            fn bitxor_assign(&mut self, rhs: bool) {
                *self ^= Self::splat(rhs);
            }
        }
    }
}

pub mod full_masks;

macro_rules! define_opaque_mask {
    {
        $(#[$attr:meta])*
        struct $name:ident<const $lanes:ident: usize>($inner_ty:ty);
    } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        pub struct $name<const $lanes: usize>($inner_ty);

        delegate_ops_to_inner! { $name }

        impl<const $lanes: usize> $name<$lanes> {
            /// Construct a mask by setting all lanes to the given value.
            pub fn splat(value: bool) -> Self {
                Self(<$inner_ty>::splat(value))
            }

            /// Tests the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn test(&self, lane: usize) -> bool {
                self.0.test(lane)
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
            #[inline]
            pub fn set(&mut self, lane: usize, value: bool) {
                self.0.set(lane, value);
            }
        }

        impl<const $lanes: usize> Copy for $name<$lanes> {}

        impl<const $lanes: usize> Clone for $name<$lanes> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const $lanes: usize> Default for $name<$lanes> {
            #[inline]
            fn default() -> Self {
                Self::splat(false)
            }
        }

        impl<const $lanes: usize> PartialEq for $name<$lanes> {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<const $lanes: usize> PartialOrd for $name<$lanes> {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<const $lanes: usize> core::fmt::Debug for $name<$lanes> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Debug::fmt(&self.0, f)
            }
        }
    };
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 8-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask8<const LANES: usize>(full_masks::SimdI8Mask<LANES>);
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 16-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask16<const LANES: usize>(full_masks::SimdI16Mask<LANES>);
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 32-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask32<const LANES: usize>(full_masks::SimdI32Mask<LANES>);
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 64-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask64<const LANES: usize>(full_masks::SimdI64Mask<LANES>);
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 128-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask128<const LANES: usize>(full_masks::SimdI128Mask<LANES>);
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` pointer-width elements.
    ///
    /// The layout of this type is unspecified.
    struct MaskSize<const LANES: usize>(full_masks::SimdIsizeMask<LANES>);
}

/// Mask-related operations using a particular mask layout.
pub trait MaskExt<Mask> {
    /// Test if each lane is equal to the corresponding lane in `other`.
    fn lanes_eq(&self, other: &Self) -> Mask;

    /// Test if each lane is not equal to the corresponding lane in `other`.
    fn lanes_ne(&self, other: &Self) -> Mask;

    /// Test if each lane is less than the corresponding lane in `other`.
    fn lanes_lt(&self, other: &Self) -> Mask;

    /// Test if each lane is greater than the corresponding lane in `other`.
    fn lanes_gt(&self, other: &Self) -> Mask;

    /// Test if each lane is less than or equal to the corresponding lane in `other`.
    fn lanes_le(&self, other: &Self) -> Mask;

    /// Test if each lane is greater than or equal to the corresponding lane in `other`.
    fn lanes_ge(&self, other: &Self) -> Mask;
}

macro_rules! implement_mask_ops {
    { $($vector:ident => $mask:ident,)* } => {
        $(
            impl<const LANES: usize> crate::$vector<LANES> {
                /// Test if each lane is equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_eq(&self, other: &Self) -> $mask<LANES> {
                    $mask(MaskExt::lanes_eq(self, other))
                }

                /// Test if each lane is not equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ne(&self, other: &Self) -> $mask<LANES> {
                    $mask(MaskExt::lanes_ne(self, other))
                }

                /// Test if each lane is less than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_lt(&self, other: &Self) -> $mask<LANES> {
                    $mask(MaskExt::lanes_lt(self, other))
                }

                /// Test if each lane is greater than the corresponding lane in `other`.
                #[inline]
                pub fn lanes_gt(&self, other: &Self) -> $mask<LANES> {
                    $mask(MaskExt::lanes_gt(self, other))
                }

                /// Test if each lane is less than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_le(&self, other: &Self) -> $mask<LANES> {
                    $mask(MaskExt::lanes_le(self, other))
                }

                /// Test if each lane is greater than or equal to the corresponding lane in `other`.
                #[inline]
                pub fn lanes_ge(&self, other: &Self) -> $mask<LANES> {
                    $mask(MaskExt::lanes_ge(self, other))
                }
            }
        )*
    }
}

implement_mask_ops! {
    SimdI8 => Mask8,
    SimdI16 => Mask16,
    SimdI32 => Mask32,
    SimdI64 => Mask64,
    SimdI128 => Mask128,
    SimdIsize => MaskSize,

    SimdU8 => Mask8,
    SimdU16 => Mask16,
    SimdU32 => Mask32,
    SimdU64 => Mask64,
    SimdU128 => Mask128,
    SimdUsize => MaskSize,

    SimdF32 => Mask32,
    SimdF64 => Mask64,
}

/// Vector of eight 8-bit masks
pub type mask8x8 = Mask8<8>;

/// Vector of 16 8-bit masks
pub type mask8x16 = Mask8<16>;

/// Vector of 32 8-bit masks
pub type mask8x32 = Mask8<32>;

/// Vector of 16 8-bit masks
pub type mask8x64 = Mask8<64>;

/// Vector of four 16-bit masks
pub type mask16x4 = Mask16<4>;

/// Vector of eight 16-bit masks
pub type mask16x8 = Mask16<8>;

/// Vector of 16 16-bit masks
pub type mask16x16 = Mask16<16>;

/// Vector of 32 16-bit masks
pub type mask16x32 = Mask32<32>;

/// Vector of two 32-bit masks
pub type mask32x2 = Mask32<2>;

/// Vector of four 32-bit masks
pub type mask32x4 = Mask32<4>;

/// Vector of eight 32-bit masks
pub type mask32x8 = Mask32<8>;

/// Vector of 16 32-bit masks
pub type mask32x16 = Mask32<16>;

/// Vector of two 64-bit masks
pub type mask64x2 = Mask64<2>;

/// Vector of four 64-bit masks
pub type mask64x4 = Mask64<4>;

/// Vector of eight 64-bit masks
pub type mask64x8 = Mask64<8>;

/// Vector of two 128-bit masks
pub type mask128x2 = Mask128<2>;

/// Vector of four 128-bit masks
pub type mask128x4 = Mask128<4>;

/// Vector of two pointer-width masks
pub type masksizex2 = MaskSize<2>;

/// Vector of four pointer-width masks
pub type masksizex4 = MaskSize<4>;

/// Vector of eight pointer-width masks
pub type masksizex8 = MaskSize<8>;
