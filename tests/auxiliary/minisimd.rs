//! Auxiliary crate for tests that need SIMD types.
//!
//! Historically the tests just made their own, but projections into simd types
//! was banned by <https://github.com/rust-lang/compiler-team/issues/838>, which
//! breaks `derive(Clone)`, so this exists to give easily-usable types that can
//! be used without copy-pasting the definitions of the helpers everywhere.
//!
//! This makes no attempt to guard against ICEs.  Using it with proper types
//! and such is your responsibility in the tests you write.

#![allow(unused)]
#![allow(non_camel_case_types)]

// The field is currently left `pub` for convenience in porting tests, many of
// which attempt to just construct it directly. That still works; it's just the
// `.0` projection that doesn't.
#[repr(simd)]
#[derive(Copy, Eq)]
pub struct Simd<T, const N: usize>(pub [T; N]);

impl<T: Copy, const N: usize> Clone for Simd<T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PartialEq, const N: usize> PartialEq for Simd<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_array() == other.as_array()
    }
}

impl<T: core::fmt::Debug, const N: usize> core::fmt::Debug for Simd<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        <[T; N] as core::fmt::Debug>::fmt(self.as_array(), f)
    }
}

impl<T, const N: usize> core::ops::Index<usize> for Simd<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.as_array()[i]
    }
}

impl<T, const N: usize> Simd<T, N> {
    pub const fn from_array(a: [T; N]) -> Self {
        Simd(a)
    }
    pub fn as_array(&self) -> &[T; N] {
        let p: *const Self = self;
        unsafe { &*p.cast::<[T; N]>() }
    }
    pub fn into_array(self) -> [T; N]
    where
        T: Copy,
    {
        *self.as_array()
    }
}

pub type u8x2 = Simd<u8, 2>;
pub type u8x4 = Simd<u8, 4>;
pub type u8x8 = Simd<u8, 8>;
pub type u8x16 = Simd<u8, 16>;
pub type u8x32 = Simd<u8, 32>;
pub type u8x64 = Simd<u8, 64>;

pub type u16x2 = Simd<u16, 2>;
pub type u16x4 = Simd<u16, 4>;
pub type u16x8 = Simd<u16, 8>;
pub type u16x16 = Simd<u16, 16>;
pub type u16x32 = Simd<u16, 32>;

pub type u32x2 = Simd<u32, 2>;
pub type u32x4 = Simd<u32, 4>;
pub type u32x8 = Simd<u32, 8>;
pub type u32x16 = Simd<u32, 16>;

pub type u64x2 = Simd<u64, 2>;
pub type u64x4 = Simd<u64, 4>;
pub type u64x8 = Simd<u64, 8>;

pub type u128x2 = Simd<u128, 2>;
pub type u128x4 = Simd<u128, 4>;

pub type i8x2 = Simd<i8, 2>;
pub type i8x4 = Simd<i8, 4>;
pub type i8x8 = Simd<i8, 8>;
pub type i8x16 = Simd<i8, 16>;
pub type i8x32 = Simd<i8, 32>;
pub type i8x64 = Simd<i8, 64>;

pub type i16x2 = Simd<i16, 2>;
pub type i16x4 = Simd<i16, 4>;
pub type i16x8 = Simd<i16, 8>;
pub type i16x16 = Simd<i16, 16>;
pub type i16x32 = Simd<i16, 32>;

pub type i32x2 = Simd<i32, 2>;
pub type i32x4 = Simd<i32, 4>;
pub type i32x8 = Simd<i32, 8>;
pub type i32x16 = Simd<i32, 16>;

pub type i64x2 = Simd<i64, 2>;
pub type i64x4 = Simd<i64, 4>;
pub type i64x8 = Simd<i64, 8>;

pub type i128x2 = Simd<i128, 2>;
pub type i128x4 = Simd<i128, 4>;

pub type f32x2 = Simd<f32, 2>;
pub type f32x4 = Simd<f32, 4>;
pub type f32x8 = Simd<f32, 8>;
pub type f32x16 = Simd<f32, 16>;

pub type f64x2 = Simd<f64, 2>;
pub type f64x4 = Simd<f64, 4>;
pub type f64x8 = Simd<f64, 8>;

// The field is currently left `pub` for convenience in porting tests, many of
// which attempt to just construct it directly. That still works; it's just the
// `.0` projection that doesn't.
#[repr(simd, packed)]
#[derive(Copy)]
pub struct PackedSimd<T, const N: usize>(pub [T; N]);

impl<T: Copy, const N: usize> Clone for PackedSimd<T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PartialEq, const N: usize> PartialEq for PackedSimd<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_array() == other.as_array()
    }
}

impl<T: core::fmt::Debug, const N: usize> core::fmt::Debug for PackedSimd<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        <[T; N] as core::fmt::Debug>::fmt(self.as_array(), f)
    }
}

impl<T, const N: usize> PackedSimd<T, N> {
    pub const fn from_array(a: [T; N]) -> Self {
        PackedSimd(a)
    }
    pub fn as_array(&self) -> &[T; N] {
        let p: *const Self = self;
        unsafe { &*p.cast::<[T; N]>() }
    }
    pub fn into_array(self) -> [T; N]
    where
        T: Copy,
    {
        *self.as_array()
    }
}
