//! Internal `#[repr(simd)]` types

#![allow(non_camel_case_types)]

macro_rules! simd_ty {
    ($id:ident [$elem_type:ty ; $len:literal]: $($param_name:ident),*) => {
        #[repr(simd)]
        #[derive(Copy, Clone)]
        pub(crate) struct $id([$elem_type; $len]);

        #[allow(clippy::use_self)]
        impl $id {
            /// A value of this type where all elements are zeroed out.
            pub(crate) const ZERO: Self = unsafe { crate::mem::zeroed() };

            #[inline(always)]
            pub(crate) const fn new($($param_name: $elem_type),*) -> Self {
                $id([$($param_name),*])
            }
            #[inline(always)]
            pub(crate) const fn from_array(elements: [$elem_type; $len]) -> Self {
                $id(elements)
            }
            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn splat(value: $elem_type) -> Self {
                #[derive(Copy, Clone)]
                #[repr(simd)]
                struct JustOne([$elem_type; 1]);
                let one = JustOne([value]);
                // SAFETY: 0 is always in-bounds because we're shuffling
                // a simd type with exactly one element.
                unsafe { simd_shuffle!(one, one, [0; $len]) }
            }

            /// Extract the element at position `index`.
            /// `index` is not a constant so this is not efficient!
            /// Use for testing only.
            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn extract(&self, index: usize) -> $elem_type {
                self.as_array()[index]
            }

            #[inline]
            pub(crate) fn as_array(&self) -> &[$elem_type; $len] {
                let simd_ptr: *const Self = self;
                let array_ptr: *const [$elem_type; $len] = simd_ptr.cast();
                // SAFETY: We can always read the prefix of a simd type as an array.
                // There might be more padding afterwards for some widths, but
                // that's not a problem for reading less than that.
                unsafe { &*array_ptr }
            }
        }

        impl core::cmp::PartialEq for $id {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_array() == other.as_array()
            }
        }

        impl core::fmt::Debug for $id {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                debug_simd_finish(f, stringify!($id), self.as_array())
            }
        }
    }
}

macro_rules! simd_m_ty {
    ($id:ident [$elem_type:ident ; $len:literal]: $($param_name:ident),*) => {
        #[repr(simd)]
        #[derive(Copy, Clone)]
        pub(crate) struct $id([$elem_type; $len]);

        #[allow(clippy::use_self)]
        impl $id {
            #[inline(always)]
            const fn bool_to_internal(x: bool) -> $elem_type {
                [0 as $elem_type, !(0 as $elem_type)][x as usize]
            }

            #[inline(always)]
            pub(crate) const fn new($($param_name: bool),*) -> Self {
                $id([$(Self::bool_to_internal($param_name)),*])
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn splat(value: bool) -> Self {
                #[derive(Copy, Clone)]
                #[repr(simd)]
                struct JustOne([$elem_type; 1]);
                let one = JustOne([Self::bool_to_internal(value)]);
                // SAFETY: 0 is always in-bounds because we're shuffling
                // a simd type with exactly one element.
                unsafe { simd_shuffle!(one, one, [0; $len]) }
            }

            #[inline]
            pub(crate) fn as_array(&self) -> &[$elem_type; $len] {
                let simd_ptr: *const Self = self;
                let array_ptr: *const [$elem_type; $len] = simd_ptr.cast();
                // SAFETY: We can always read the prefix of a simd type as an array.
                // There might be more padding afterwards for some widths, but
                // that's not a problem for reading less than that.
                unsafe { &*array_ptr }
            }
        }

        impl core::cmp::PartialEq for $id {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_array() == other.as_array()
            }
        }

        impl core::fmt::Debug for $id {
            #[inline]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                debug_simd_finish(f, stringify!($id), self.as_array())
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
