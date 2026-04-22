//! Internal `#[repr(simd)]` types

#![allow(non_camel_case_types)]

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const unsafe fn simd_imax<T: Copy>(a: T, b: T) -> T {
    let mask: T = crate::intrinsics::simd::simd_gt(a, b);
    crate::intrinsics::simd::simd_select(mask, a, b)
}

#[inline(always)]
#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
pub(crate) const unsafe fn simd_imin<T: Copy>(a: T, b: T) -> T {
    let mask: T = crate::intrinsics::simd::simd_lt(a, b);
    crate::intrinsics::simd::simd_select(mask, a, b)
}

/// SAFETY: All bits patterns must be valid
pub(crate) unsafe trait SimdElement:
    Copy + const PartialEq + crate::fmt::Debug
{
    // SAFETY: all bits patterns of types implementing this trait must be valid
    const ZERO: Self = unsafe { crate::mem::zeroed() };
}

unsafe impl SimdElement for u8 {}
unsafe impl SimdElement for u16 {}
unsafe impl SimdElement for u32 {}
unsafe impl SimdElement for u64 {}

unsafe impl SimdElement for i8 {}
unsafe impl SimdElement for i16 {}
unsafe impl SimdElement for i32 {}
unsafe impl SimdElement for i64 {}

unsafe impl SimdElement for f16 {}
unsafe impl SimdElement for f32 {}
unsafe impl SimdElement for f64 {}

#[repr(simd)]
#[derive(Copy)]
pub(crate) struct Simd<T: SimdElement, const N: usize>([T; N]);

impl<T: SimdElement, const N: usize> Simd<T, N> {
    /// A value of this type where all elements are zeroed out.
    pub(crate) const ZERO: Self = Self::splat(T::ZERO);

    #[inline(always)]
    pub(crate) const fn from_array(elements: [T; N]) -> Self {
        Self(elements)
    }

    #[inline]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub(crate) const fn splat(value: T) -> Self {
        unsafe { crate::intrinsics::simd::simd_splat(value) }
    }

    /// Extract the element at position `index`. Note that `index` is not a constant so this
    /// operation is not efficient on most platforms. Use for testing only.
    #[inline]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub(crate) const fn extract_dyn(&self, index: usize) -> T {
        assert!(index < N);
        // SAFETY: self is a vector, T its element type.
        unsafe { crate::intrinsics::simd::simd_extract_dyn(*self, index as u32) }
    }

    #[inline]
    pub(crate) const fn as_array(&self) -> &[T; N] {
        let simd_ptr: *const Self = self;
        let array_ptr: *const [T; N] = simd_ptr.cast();
        // SAFETY: We can always read the prefix of a simd type as an array.
        // There might be more padding afterwards for some widths, but
        // that's not a problem for reading less than that.
        unsafe { &*array_ptr }
    }
}

// `#[derive(Clone)]` causes ICE "Projecting into SIMD type core_arch::simd::Simd is banned by MCP#838"
impl<T: SimdElement, const N: usize> Clone for Simd<T, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
impl<T: SimdElement, const N: usize> const crate::cmp::PartialEq for Simd<T, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_array() == other.as_array()
    }
}

impl<T: SimdElement, const N: usize> crate::fmt::Debug for Simd<T, N> {
    #[inline]
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        debug_simd_finish(f, "Simd", self.as_array())
    }
}

impl<T: SimdElement> Simd<T, 1> {
    #[inline]
    pub(crate) const fn new(x0: T) -> Self {
        Self([x0])
    }
}

impl<T: SimdElement> Simd<T, 2> {
    #[inline]
    pub(crate) const fn new(x0: T, x1: T) -> Self {
        Self([x0, x1])
    }
}

impl<T: SimdElement> Simd<T, 4> {
    #[inline]
    pub(crate) const fn new(x0: T, x1: T, x2: T, x3: T) -> Self {
        Self([x0, x1, x2, x3])
    }
}

impl<T: SimdElement> Simd<T, 8> {
    #[inline]
    pub(crate) const fn new(x0: T, x1: T, x2: T, x3: T, x4: T, x5: T, x6: T, x7: T) -> Self {
        Self([x0, x1, x2, x3, x4, x5, x6, x7])
    }
}

impl<T: SimdElement> Simd<T, 16> {
    #[inline]
    pub(crate) const fn new(
        x0: T,
        x1: T,
        x2: T,
        x3: T,
        x4: T,
        x5: T,
        x6: T,
        x7: T,
        x8: T,
        x9: T,
        x10: T,
        x11: T,
        x12: T,
        x13: T,
        x14: T,
        x15: T,
    ) -> Self {
        Self([
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
        ])
    }
}

impl<T: SimdElement> Simd<T, 32> {
    #[inline]
    pub(crate) const fn new(
        x0: T,
        x1: T,
        x2: T,
        x3: T,
        x4: T,
        x5: T,
        x6: T,
        x7: T,
        x8: T,
        x9: T,
        x10: T,
        x11: T,
        x12: T,
        x13: T,
        x14: T,
        x15: T,
        x16: T,
        x17: T,
        x18: T,
        x19: T,
        x20: T,
        x21: T,
        x22: T,
        x23: T,
        x24: T,
        x25: T,
        x26: T,
        x27: T,
        x28: T,
        x29: T,
        x30: T,
        x31: T,
    ) -> Self {
        Self([
            x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18,
            x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31,
        ])
    }
}

impl<const N: usize> Simd<f16, N> {
    #[inline]
    pub(crate) const fn to_bits(self) -> Simd<u16, N> {
        assert!(size_of::<Self>() == size_of::<Simd<u16, N>>());
        unsafe { crate::mem::transmute_copy(&self) }
    }

    #[inline]
    pub(crate) const fn from_bits(bits: Simd<u16, N>) -> Self {
        assert!(size_of::<Self>() == size_of::<Simd<u16, N>>());
        unsafe { crate::mem::transmute_copy(&bits) }
    }
}

impl<const N: usize> Simd<f32, N> {
    #[inline]
    pub(crate) const fn to_bits(self) -> Simd<u32, N> {
        assert!(size_of::<Self>() == size_of::<Simd<u32, N>>());
        unsafe { crate::mem::transmute_copy(&self) }
    }

    #[inline]
    pub(crate) const fn from_bits(bits: Simd<u32, N>) -> Self {
        assert!(size_of::<Self>() == size_of::<Simd<u32, N>>());
        unsafe { crate::mem::transmute_copy(&bits) }
    }
}

impl<const N: usize> Simd<f64, N> {
    #[inline]
    pub(crate) const fn to_bits(self) -> Simd<u64, N> {
        assert!(size_of::<Self>() == size_of::<Simd<u64, N>>());
        unsafe { crate::mem::transmute_copy(&self) }
    }

    #[inline]
    pub(crate) const fn from_bits(bits: Simd<u64, N>) -> Self {
        assert!(size_of::<Self>() == size_of::<Simd<u64, N>>());
        unsafe { crate::mem::transmute_copy(&bits) }
    }
}

#[repr(simd)]
#[derive(Copy)]
pub(crate) struct SimdM<T: SimdElement, const N: usize>([T; N]);

impl<T: SimdElement, const N: usize> SimdM<T, N> {
    #[inline(always)]
    const fn bool_to_internal(x: bool) -> T {
        // SAFETY: `T` implements `SimdElement`, so all bit patterns are valid.
        let ones = const {
            // Ideally, this would be `transmute([0xFFu8; size_of::<T>()])`, but
            // `size_of::<T>()` is not allowed to use a generic parameter there.
            let mut r = crate::mem::MaybeUninit::<T>::uninit();
            let mut i = 0;
            while i < crate::mem::size_of::<T>() {
                r.as_bytes_mut()[i] = crate::mem::MaybeUninit::new(0xFF);
                i += 1;
            }
            unsafe { r.assume_init() }
        };
        [T::ZERO, ones][x as usize]
    }

    #[inline]
    pub(crate) const fn from_array(elements: [bool; N]) -> Self {
        let mut internal = [T::ZERO; N];
        let mut i = 0;
        while i < N {
            internal[i] = Self::bool_to_internal(elements[i]);
            i += 1;
        }
        Self(internal)
    }

    #[inline]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub(crate) const fn splat(value: bool) -> Self {
        unsafe { crate::intrinsics::simd::simd_splat(Self::bool_to_internal(value)) }
    }

    #[inline]
    pub(crate) const fn as_array(&self) -> &[T; N] {
        let simd_ptr: *const Self = self;
        let array_ptr: *const [T; N] = simd_ptr.cast();
        // SAFETY: We can always read the prefix of a simd type as an array.
        // There might be more padding afterwards for some widths, but
        // that's not a problem for reading less than that.
        unsafe { &*array_ptr }
    }
}

// `#[derive(Clone)]` causes ICE "Projecting into SIMD type core_arch::simd::SimdM is banned by MCP#838"
impl<T: SimdElement, const N: usize> Clone for SimdM<T, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
impl<T: SimdElement, const N: usize> const crate::cmp::PartialEq for SimdM<T, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_array() == other.as_array()
    }
}

impl<T: SimdElement, const N: usize> crate::fmt::Debug for SimdM<T, N> {
    #[inline]
    fn fmt(&self, f: &mut crate::fmt::Formatter<'_>) -> crate::fmt::Result {
        debug_simd_finish(f, "SimdM", self.as_array())
    }
}

// 16-bit wide types:

pub(crate) type u8x2 = Simd<u8, 2>;
pub(crate) type i8x2 = Simd<i8, 2>;

// 32-bit wide types:

pub(crate) type u8x4 = Simd<u8, 4>;
pub(crate) type u16x2 = Simd<u16, 2>;

pub(crate) type i8x4 = Simd<i8, 4>;
pub(crate) type i16x2 = Simd<i16, 2>;

// 64-bit wide types:

pub(crate) type u8x8 = Simd<u8, 8>;
pub(crate) type u16x4 = Simd<u16, 4>;
pub(crate) type u32x2 = Simd<u32, 2>;
pub(crate) type u64x1 = Simd<u64, 1>;

pub(crate) type i8x8 = Simd<i8, 8>;
pub(crate) type i16x4 = Simd<i16, 4>;
pub(crate) type i32x2 = Simd<i32, 2>;
pub(crate) type i64x1 = Simd<i64, 1>;

pub(crate) type f16x4 = Simd<f16, 4>;
pub(crate) type f32x2 = Simd<f32, 2>;
pub(crate) type f64x1 = Simd<f64, 1>;

// 128-bit wide types:

pub(crate) type u8x16 = Simd<u8, 16>;
pub(crate) type u16x8 = Simd<u16, 8>;
pub(crate) type u32x4 = Simd<u32, 4>;
pub(crate) type u64x2 = Simd<u64, 2>;

pub(crate) type i8x16 = Simd<i8, 16>;
pub(crate) type i16x8 = Simd<i16, 8>;
pub(crate) type i32x4 = Simd<i32, 4>;
pub(crate) type i64x2 = Simd<i64, 2>;

pub(crate) type f16x8 = Simd<f16, 8>;
pub(crate) type f32x4 = Simd<f32, 4>;
pub(crate) type f64x2 = Simd<f64, 2>;

pub(crate) type m8x16 = SimdM<i8, 16>;
pub(crate) type m16x8 = SimdM<i16, 8>;
pub(crate) type m32x4 = SimdM<i32, 4>;
pub(crate) type m64x2 = SimdM<i64, 2>;

// 256-bit wide types:

pub(crate) type u8x32 = Simd<u8, 32>;
pub(crate) type u16x16 = Simd<u16, 16>;
pub(crate) type u32x8 = Simd<u32, 8>;
pub(crate) type u64x4 = Simd<u64, 4>;

pub(crate) type i8x32 = Simd<i8, 32>;
pub(crate) type i16x16 = Simd<i16, 16>;
pub(crate) type i32x8 = Simd<i32, 8>;
pub(crate) type i64x4 = Simd<i64, 4>;

pub(crate) type f16x16 = Simd<f16, 16>;
pub(crate) type f32x8 = Simd<f32, 8>;
pub(crate) type f64x4 = Simd<f64, 4>;

pub(crate) type m8x32 = SimdM<i8, 32>;
pub(crate) type m16x16 = SimdM<i16, 16>;
pub(crate) type m32x8 = SimdM<i32, 8>;

// 512-bit wide types:

pub(crate) type u8x64 = Simd<u8, 64>;
pub(crate) type u16x32 = Simd<u16, 32>;
pub(crate) type u32x16 = Simd<u32, 16>;
pub(crate) type u64x8 = Simd<u64, 8>;

pub(crate) type i8x64 = Simd<i8, 64>;
pub(crate) type i16x32 = Simd<i16, 32>;
pub(crate) type i32x16 = Simd<i32, 16>;
pub(crate) type i64x8 = Simd<i64, 8>;

pub(crate) type f16x32 = Simd<f16, 32>;
pub(crate) type f32x16 = Simd<f32, 16>;
pub(crate) type f64x8 = Simd<f64, 8>;

// 1024-bit wide types:

pub(crate) type u16x64 = Simd<u16, 64>;
pub(crate) type u32x32 = Simd<u32, 32>;

pub(crate) type i32x32 = Simd<i32, 32>;

/// Used to continue `Debug`ging SIMD types as `MySimd(1, 2, 3, 4)`, as they
/// were before moving to array-based simd.
#[inline]
pub(crate) fn debug_simd_finish<T: crate::fmt::Debug, const N: usize>(
    formatter: &mut crate::fmt::Formatter<'_>,
    type_name: &str,
    array: &[T; N],
) -> crate::fmt::Result {
    crate::fmt::Formatter::debug_tuple_fields_finish(
        formatter,
        type_name,
        &crate::array::from_fn::<&dyn crate::fmt::Debug, N, _>(|i| &array[i]),
    )
}
