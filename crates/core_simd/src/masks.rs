//! Types and traits associated with masking lanes of vectors.
//! Types representing
#![allow(non_camel_case_types)]

#[cfg_attr(
    not(all(target_arch = "x86_64", target_feature = "avx512f")),
    path = "masks/full_masks.rs"
)]
#[cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    path = "masks/bitmask.rs"
)]
mod mask_impl;

use crate::{LaneCount, Simd, SimdElement, SupportedLaneCount};

/// Marker trait for types that may be used as SIMD mask elements.
pub unsafe trait MaskElement: SimdElement {
    #[doc(hidden)]
    fn valid<const LANES: usize>(values: Simd<Self, LANES>) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount;

    #[doc(hidden)]
    fn eq(self, other: Self) -> bool;

    #[doc(hidden)]
    const TRUE: Self;

    #[doc(hidden)]
    const FALSE: Self;
}

macro_rules! impl_element {
    { $ty:ty } => {
        unsafe impl MaskElement for $ty {
            fn valid<const LANES: usize>(value: Simd<Self, LANES>) -> bool
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                (value.lanes_eq(Simd::splat(0)) | value.lanes_eq(Simd::splat(-1))).all()
            }

            fn eq(self, other: Self) -> bool { self == other }

            const TRUE: Self = -1;
            const FALSE: Self = 0;
        }
    }
}

impl_element! { i8 }
impl_element! { i16 }
impl_element! { i32 }
impl_element! { i64 }
impl_element! { isize }

/// A SIMD vector mask for `LANES` elements of width specified by `Element`.
///
/// The layout of this type is unspecified.
#[repr(transparent)]
pub struct Mask<Element, const LANES: usize>(mask_impl::Mask<Element, LANES>)
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount;

impl<Element, const LANES: usize> Copy for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
}

impl<Element, const LANES: usize> Clone for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Element, const LANES: usize> Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Construct a mask by setting all lanes to the given value.
    pub fn splat(value: bool) -> Self {
        Self(mask_impl::Mask::splat(value))
    }

    /// Converts an array to a SIMD vector.
    pub fn from_array(array: [bool; LANES]) -> Self {
        let mut vector = Self::splat(false);
        for (i, v) in array.iter().enumerate() {
            vector.set(i, *v);
        }
        vector
    }

    /// Converts a SIMD vector to an array.
    pub fn to_array(self) -> [bool; LANES] {
        let mut array = [false; LANES];
        for (i, v) in array.iter_mut().enumerate() {
            *v = self.test(i);
        }
        array
    }

    /// Converts a vector of integers to a mask, where 0 represents `false` and -1
    /// represents `true`.
    ///
    /// # Safety
    /// All lanes must be either 0 or -1.
    #[inline]
    pub unsafe fn from_int_unchecked(value: Simd<Element, LANES>) -> Self {
        Self(mask_impl::Mask::from_int_unchecked(value))
    }

    /// Converts a vector of integers to a mask, where 0 represents `false` and -1
    /// represents `true`.
    ///
    /// # Panics
    /// Panics if any lane is not 0 or -1.
    #[inline]
    pub fn from_int(value: Simd<Element, LANES>) -> Self {
        assert!(Element::valid(value), "all values must be either 0 or -1",);
        unsafe { Self::from_int_unchecked(value) }
    }

    /// Converts the mask to a vector of integers, where 0 represents `false` and -1
    /// represents `true`.
    #[inline]
    pub fn to_int(self) -> Simd<Element, LANES> {
        self.0.to_int()
    }

    /// Tests the value of the specified lane.
    ///
    /// # Safety
    /// `lane` must be less than `LANES`.
    #[inline]
    pub unsafe fn test_unchecked(&self, lane: usize) -> bool {
        self.0.test_unchecked(lane)
    }

    /// Tests the value of the specified lane.
    ///
    /// # Panics
    /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
    #[inline]
    pub fn test(&self, lane: usize) -> bool {
        assert!(lane < LANES, "lane index out of range");
        unsafe { self.test_unchecked(lane) }
    }

    /// Sets the value of the specified lane.
    ///
    /// # Safety
    /// `lane` must be less than `LANES`.
    #[inline]
    pub unsafe fn set_unchecked(&mut self, lane: usize, value: bool) {
        self.0.set_unchecked(lane, value);
    }

    /// Sets the value of the specified lane.
    ///
    /// # Panics
    /// Panics if `lane` is greater than or equal to the number of lanes in the vector.
    #[inline]
    pub fn set(&mut self, lane: usize, value: bool) {
        assert!(lane < LANES, "lane index out of range");
        unsafe {
            self.set_unchecked(lane, value);
        }
    }

    /// Convert this mask to a bitmask, with one bit set per lane.
    pub fn to_bitmask(self) -> [u8; LaneCount::<LANES>::BITMASK_LEN] {
        self.0.to_bitmask()
    }

    /// Convert a bitmask to a mask.
    pub fn from_bitmask(bitmask: [u8; LaneCount::<LANES>::BITMASK_LEN]) -> Self {
        Self(mask_impl::Mask::from_bitmask(bitmask))
    }

    /// Returns true if any lane is set, or false otherwise.
    #[inline]
    pub fn any(self) -> bool {
        self.0.any()
    }

    /// Returns true if all lanes are set, or false otherwise.
    #[inline]
    pub fn all(self) -> bool {
        self.0.all()
    }
}

// vector/array conversion
impl<Element, const LANES: usize> From<[bool; LANES]> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn from(array: [bool; LANES]) -> Self {
        Self::from_array(array)
    }
}

impl<Element, const LANES: usize> From<Mask<Element, LANES>> for [bool; LANES]
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn from(vector: Mask<Element, LANES>) -> Self {
        vector.to_array()
    }
}

impl<Element, const LANES: usize> Default for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn default() -> Self {
        Self::splat(false)
    }
}

impl<Element, const LANES: usize> PartialEq for Mask<Element, LANES>
where
    Element: MaskElement + PartialEq,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<Element, const LANES: usize> PartialOrd for Mask<Element, LANES>
where
    Element: MaskElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<Element, const LANES: usize> core::fmt::Debug for Mask<Element, LANES>
where
    Element: MaskElement + core::fmt::Debug,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_list()
            .entries((0..LANES).map(|lane| self.test(lane)))
            .finish()
    }
}

impl<Element, const LANES: usize> core::ops::BitAnd for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl<Element, const LANES: usize> core::ops::BitAnd<bool> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: bool) -> Self {
        self & Self::splat(rhs)
    }
}

impl<Element, const LANES: usize> core::ops::BitAnd<Mask<Element, LANES>> for bool
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Mask<Element, LANES>;
    #[inline]
    fn bitand(self, rhs: Mask<Element, LANES>) -> Mask<Element, LANES> {
        Mask::splat(self) & rhs
    }
}

impl<Element, const LANES: usize> core::ops::BitOr for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl<Element, const LANES: usize> core::ops::BitOr<bool> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: bool) -> Self {
        self | Self::splat(rhs)
    }
}

impl<Element, const LANES: usize> core::ops::BitOr<Mask<Element, LANES>> for bool
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Mask<Element, LANES>;
    #[inline]
    fn bitor(self, rhs: Mask<Element, LANES>) -> Mask<Element, LANES> {
        Mask::splat(self) | rhs
    }
}

impl<Element, const LANES: usize> core::ops::BitXor for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl<Element, const LANES: usize> core::ops::BitXor<bool> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    #[inline]
    fn bitxor(self, rhs: bool) -> Self::Output {
        self ^ Self::splat(rhs)
    }
}

impl<Element, const LANES: usize> core::ops::BitXor<Mask<Element, LANES>> for bool
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Mask<Element, LANES>;
    #[inline]
    fn bitxor(self, rhs: Mask<Element, LANES>) -> Self::Output {
        Mask::splat(self) ^ rhs
    }
}

impl<Element, const LANES: usize> core::ops::Not for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Mask<Element, LANES>;
    #[inline]
    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl<Element, const LANES: usize> core::ops::BitAndAssign for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 = self.0 & rhs.0;
    }
}

impl<Element, const LANES: usize> core::ops::BitAndAssign<bool> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn bitand_assign(&mut self, rhs: bool) {
        *self &= Self::splat(rhs);
    }
}

impl<Element, const LANES: usize> core::ops::BitOrAssign for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 = self.0 | rhs.0;
    }
}

impl<Element, const LANES: usize> core::ops::BitOrAssign<bool> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn bitor_assign(&mut self, rhs: bool) {
        *self |= Self::splat(rhs);
    }
}

impl<Element, const LANES: usize> core::ops::BitXorAssign for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 = self.0 ^ rhs.0;
    }
}

impl<Element, const LANES: usize> core::ops::BitXorAssign<bool> for Mask<Element, LANES>
where
    Element: MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline]
    fn bitxor_assign(&mut self, rhs: bool) {
        *self ^= Self::splat(rhs);
    }
}

/// A SIMD mask of `LANES` 8-bit values.
pub type Mask8<const LANES: usize> = Mask<i8, LANES>;

/// A SIMD mask of `LANES` 16-bit values.
pub type Mask16<const LANES: usize> = Mask<i16, LANES>;

/// A SIMD mask of `LANES` 32-bit values.
pub type Mask32<const LANES: usize> = Mask<i32, LANES>;

/// A SIMD mask of `LANES` 64-bit values.
pub type Mask64<const LANES: usize> = Mask<i64, LANES>;

/// A SIMD mask of `LANES` pointer-width values.
pub type MaskSize<const LANES: usize> = Mask<isize, LANES>;

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

/// Vector of two pointer-width masks
pub type masksizex2 = MaskSize<2>;

/// Vector of four pointer-width masks
pub type masksizex4 = MaskSize<4>;

/// Vector of eight pointer-width masks
pub type masksizex8 = MaskSize<8>;

macro_rules! impl_from {
    { $from:ident ($from_inner:ident) => $($to:ident ($to_inner:ident)),* } => {
        $(
        impl<const LANES: usize> From<$from<LANES>> for $to<LANES>
        where
            crate::LaneCount<LANES>: crate::SupportedLaneCount,
        {
            fn from(value: $from<LANES>) -> Self {
                Self(value.0.convert())
            }
        }
        )*
    }
}
impl_from! { Mask8 (SimdI8) => Mask16 (SimdI16), Mask32 (SimdI32), Mask64 (SimdI64), MaskSize (SimdIsize) }
impl_from! { Mask16 (SimdI16) => Mask32 (SimdI32), Mask64 (SimdI64), MaskSize (SimdIsize), Mask8 (SimdI8) }
impl_from! { Mask32 (SimdI32) => Mask64 (SimdI64), MaskSize (SimdIsize), Mask8 (SimdI8), Mask16 (SimdI16) }
impl_from! { Mask64 (SimdI64) => MaskSize (SimdIsize), Mask8 (SimdI8), Mask16 (SimdI16), Mask32 (SimdI32) }
impl_from! { MaskSize (SimdIsize) => Mask8 (SimdI8), Mask16 (SimdI16), Mask32 (SimdI32), Mask64 (SimdI64) }
