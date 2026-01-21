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
    // SAFETY: `T` implements `SimdElement`, so it is zeroable.
    pub(crate) const ZERO: Self = unsafe { crate::mem::zeroed() };

    #[inline(always)]
    pub(crate) const fn from_array(elements: [T; N]) -> Self {
        Self(elements)
    }

    // FIXME: Workaround rust@60637
    #[inline(always)]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub(crate) const fn splat(value: T) -> Self {
        let one = Simd([value]);
        // SAFETY: 0 is always in-bounds because we're shuffling
        // a simd type with exactly one element.
        unsafe { simd_shuffle!(one, one, [0; N]) }
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

macro_rules! simd_ty {
    ($id:ident [$elem_type:ty ; $len:literal]: $($param_name:ident),*) => {
        pub(crate) type $id = Simd<$elem_type, $len>;

        impl $id {
            #[inline(always)]
            pub(crate) const fn new($($param_name: $elem_type),*) -> Self {
                Self([$($param_name),*])
            }
        }
    }
}

#[repr(simd)]
#[derive(Copy)]
pub(crate) struct SimdM<T: SimdElement, const N: usize>([T; N]);

impl<T: SimdElement, const N: usize> SimdM<T, N> {
    #[inline(always)]
    const fn bool_to_internal(x: bool) -> T {
        // SAFETY: `T` implements `SimdElement`, so all bit patterns are valid.
        let zeros = const { unsafe { crate::mem::zeroed::<T>() } };
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
        [zeros, ones][x as usize]
    }

    // FIXME: Workaround rust@60637
    #[inline(always)]
    #[rustc_const_unstable(feature = "stdarch_const_helpers", issue = "none")]
    pub(crate) const fn splat(value: bool) -> Self {
        let one = SimdM([Self::bool_to_internal(value)]);
        // SAFETY: 0 is always in-bounds because we're shuffling
        // a simd type with exactly one element.
        unsafe { simd_shuffle!(one, one, [0; N]) }
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

macro_rules! simd_m_ty {
    ($id:ident [$elem_type:ident ; $len:literal]: $($param_name:ident),*) => {
        pub(crate) type $id  = SimdM<$elem_type, $len>;

        impl $id {
            #[inline(always)]
            pub(crate) const fn new($($param_name: bool),*) -> Self {
                Self([$(Self::bool_to_internal($param_name)),*])
            }
        }
    }
}

// 16-bit wide types:

simd_ty!(u8x2[u8;2]: x0, x1);
simd_ty!(i8x2[i8;2]: x0, x1);

// 32-bit wide types:

simd_ty!(u8x4[u8;4]: x0, x1, x2, x3);
simd_ty!(u16x2[u16;2]: x0, x1);

simd_ty!(i8x4[i8;4]: x0, x1, x2, x3);
simd_ty!(i16x2[i16;2]: x0, x1);

// 64-bit wide types:

simd_ty!(
    u8x8[u8;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(u16x4[u16;4]: x0, x1, x2, x3);
simd_ty!(u32x2[u32;2]: x0, x1);
simd_ty!(u64x1[u64;1]: x1);

simd_ty!(
    i8x8[i8;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(i16x4[i16;4]: x0, x1, x2, x3);
simd_ty!(i32x2[i32;2]: x0, x1);
simd_ty!(i64x1[i64;1]: x1);

simd_ty!(f32x2[f32;2]: x0, x1);
simd_ty!(f64x1[f64;1]: x1);

// 128-bit wide types:

simd_ty!(
    u8x16[u8;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_ty!(
    u16x8[u16;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(u32x4[u32;4]: x0, x1, x2, x3);
simd_ty!(u64x2[u64;2]: x0, x1);

simd_ty!(
    i8x16[i8;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_ty!(
    i16x8[i16;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(i32x4[i32;4]: x0, x1, x2, x3);
simd_ty!(i64x2[i64;2]: x0, x1);

simd_ty!(f16x4[f16;4]: x0, x1, x2, x3);

simd_ty!(
    f16x8[f16;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(f32x4[f32;4]: x0, x1, x2, x3);
simd_ty!(f64x2[f64;2]: x0, x1);

simd_m_ty!(
    m8x16[i8;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_m_ty!(
    m16x8[i16;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_m_ty!(m32x4[i32;4]: x0, x1, x2, x3);
simd_m_ty!(m64x2[i64;2]: x0, x1);

// 256-bit wide types:

simd_ty!(
    u8x32[u8;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);
simd_ty!(
    u16x16[u16;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_ty!(
    u32x8[u32;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(u64x4[u64;4]: x0, x1, x2, x3);

simd_ty!(
    i8x32[i8;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);
simd_ty!(
    i16x16[i16;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_ty!(
    i32x8[i32;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(i64x4[i64;4]: x0, x1, x2, x3);

simd_ty!(
    f16x16[f16;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_ty!(
    f32x8[f32;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);
simd_ty!(f64x4[f64;4]: x0, x1, x2, x3);

simd_m_ty!(
    m8x32[i8;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);
simd_m_ty!(
    m16x16[i16;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);
simd_m_ty!(
    m32x8[i32;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);

// 512-bit wide types:

simd_ty!(
    i8x64[i8;64]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31,
    x32,
    x33,
    x34,
    x35,
    x36,
    x37,
    x38,
    x39,
    x40,
    x41,
    x42,
    x43,
    x44,
    x45,
    x46,
    x47,
    x48,
    x49,
    x50,
    x51,
    x52,
    x53,
    x54,
    x55,
    x56,
    x57,
    x58,
    x59,
    x60,
    x61,
    x62,
    x63
);

simd_ty!(
    u8x64[u8;64]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31,
    x32,
    x33,
    x34,
    x35,
    x36,
    x37,
    x38,
    x39,
    x40,
    x41,
    x42,
    x43,
    x44,
    x45,
    x46,
    x47,
    x48,
    x49,
    x50,
    x51,
    x52,
    x53,
    x54,
    x55,
    x56,
    x57,
    x58,
    x59,
    x60,
    x61,
    x62,
    x63
);

simd_ty!(
    i16x32[i16;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);

simd_ty!(
    u16x32[u16;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);

simd_ty!(
    i32x16[i32;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);

simd_ty!(
    u32x16[u32;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);

simd_ty!(
    f16x32[f16;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);
simd_ty!(
    f32x16[f32;16]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15
);

simd_ty!(
    i64x8[i64;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);

simd_ty!(
    u64x8[u64;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);

simd_ty!(
    f64x8[f64;8]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7
);

// 1024-bit wide types:
simd_ty!(
    u16x64[u16;64]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31,
    x32,
    x33,
    x34,
    x35,
    x36,
    x37,
    x38,
    x39,
    x40,
    x41,
    x42,
    x43,
    x44,
    x45,
    x46,
    x47,
    x48,
    x49,
    x50,
    x51,
    x52,
    x53,
    x54,
    x55,
    x56,
    x57,
    x58,
    x59,
    x60,
    x61,
    x62,
    x63
);
simd_ty!(
    i32x32[i32;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);
simd_ty!(
    u32x32[u32;32]:
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
    x16,
    x17,
    x18,
    x19,
    x20,
    x21,
    x22,
    x23,
    x24,
    x25,
    x26,
    x27,
    x28,
    x29,
    x30,
    x31
);

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
