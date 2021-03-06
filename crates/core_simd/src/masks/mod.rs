//! Types and traits associated with masking lanes of vectors.
#![allow(non_camel_case_types)]

mod full_masks;
pub use full_masks::*;

mod bitmask;
pub use bitmask::*;

use crate::LanesAtMost64;

macro_rules! define_opaque_mask {
    {
        $(#[$attr:meta])*
        struct $name:ident<const $lanes:ident: usize>($inner_ty:ty);
        @bits $bits_ty:ty
    } => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        pub struct $name<const $lanes: usize>($inner_ty) where $bits_ty: LanesAtMost64;

        impl<const $lanes: usize> $name<$lanes>
        where
            $bits_ty: LanesAtMost64
        {
            /// Construct a mask by setting all lanes to the given value.
            pub fn splat(value: bool) -> Self {
                Self(<$inner_ty>::splat(value))
            }

            /// Converts an array to a SIMD vector.
            pub fn from_array(array: [bool; LANES]) -> Self {
                let mut vector = Self::splat(false);
                let mut i = 0;
                while i < $lanes {
                    vector.set(i, array[i]);
                    i += 1;
                }
                vector
            }

            /// Converts a SIMD vector to an array.
            pub fn to_array(self) -> [bool; LANES] {
                let mut array = [false; LANES];
                let mut i = 0;
                while i < $lanes {
                    array[i] = self.test(i);
                    i += 1;
                }
                array
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

        impl<const $lanes: usize> From<BitMask<$lanes>> for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
            BitMask<$lanes>: LanesAtMost64,
        {
            fn from(value: BitMask<$lanes>) -> Self {
                Self(value.into())
            }
        }

        impl<const $lanes: usize> From<$name<$lanes>> for crate::BitMask<$lanes>
        where
            $bits_ty: LanesAtMost64,
            BitMask<$lanes>: LanesAtMost64,
        {
            fn from(value: $name<$lanes>) -> Self {
                value.0.into()
            }
        }

        impl<const $lanes: usize> From<$inner_ty> for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
        {
            fn from(value: $inner_ty) -> Self {
                Self(value)
            }
        }

        impl<const $lanes: usize> From<$name<$lanes>> for $inner_ty
        where
            $bits_ty: LanesAtMost64,
        {
            fn from(value: $name<$lanes>) -> Self {
                value.0
            }
        }

        // vector/array conversion
        impl<const $lanes: usize> From<[bool; $lanes]> for $name<$lanes> where $bits_ty: crate::LanesAtMost64 {
            fn from(array: [bool; $lanes]) -> Self {
                Self::from_array(array)
            }
        }

        impl <const $lanes: usize> From<$name<$lanes>> for [bool; $lanes] where $bits_ty: crate::LanesAtMost64 {
            fn from(vector: $name<$lanes>) -> Self {
                vector.to_array()
            }
        }

        impl<const $lanes: usize> Copy for $name<$lanes>
        where
            $inner_ty: Copy,
            $bits_ty: LanesAtMost64,
        {}

        impl<const $lanes: usize> Clone for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<const $lanes: usize> Default for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn default() -> Self {
                Self::splat(false)
            }
        }

        impl<const $lanes: usize> PartialEq for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }

        impl<const $lanes: usize> PartialOrd for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl<const $lanes: usize> core::fmt::Debug for $name<$lanes>
        where
            $bits_ty: LanesAtMost64,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<bool> for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: bool) -> Self {
                self & Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitAnd<$name<LANES>> for bool
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitand(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) & rhs
            }
        }

        impl<const LANES: usize> core::ops::BitOr for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<bool> for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: bool) -> Self {
                self | Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitOr<$name<LANES>> for bool
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitor(self, rhs: $name<LANES>) -> $name<LANES> {
                $name::<LANES>::splat(self) | rhs
            }
        }

        impl<const LANES: usize> core::ops::BitXor for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<bool> for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: bool) -> Self::Output {
                self ^ Self::splat(rhs)
            }
        }

        impl<const LANES: usize> core::ops::BitXor<$name<LANES>> for bool
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = $name<LANES>;
            #[inline]
            fn bitxor(self, rhs: $name<LANES>) -> Self::Output {
                $name::<LANES>::splat(self) ^ rhs
            }
        }

        impl<const LANES: usize> core::ops::Not for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            type Output = $name<LANES>;
            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitAndAssign<bool> for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn bitand_assign(&mut self, rhs: bool) {
                *self &= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitOrAssign<bool> for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn bitor_assign(&mut self, rhs: bool) {
                *self |= Self::splat(rhs);
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }

        impl<const LANES: usize> core::ops::BitXorAssign<bool> for $name<LANES>
        where
            $bits_ty: LanesAtMost64,
        {
            #[inline]
            fn bitxor_assign(&mut self, rhs: bool) {
                *self ^= Self::splat(rhs);
            }
        }
    };
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 8-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask8<const LANES: usize>(SimdMask8<LANES>);
    @bits crate::SimdI8<LANES>
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 16-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask16<const LANES: usize>(SimdMask16<LANES>);
    @bits crate::SimdI16<LANES>
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 32-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask32<const LANES: usize>(SimdMask32<LANES>);
    @bits crate::SimdI32<LANES>
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 64-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask64<const LANES: usize>(SimdMask64<LANES>);
    @bits crate::SimdI64<LANES>
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` 128-bit elements.
    ///
    /// The layout of this type is unspecified.
    struct Mask128<const LANES: usize>(SimdMask128<LANES>);
    @bits crate::SimdI128<LANES>
}

define_opaque_mask! {
    /// Mask for vectors with `LANES` pointer-width elements.
    ///
    /// The layout of this type is unspecified.
    struct MaskSize<const LANES: usize>(SimdMaskSize<LANES>);
    @bits crate::SimdIsize<LANES>
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
