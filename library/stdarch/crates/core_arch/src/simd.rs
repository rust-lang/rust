//! Internal `#[repr(simd)]` types

#![rustfmt::skip]
#![allow(non_camel_case_types)]

macro_rules! simd_ty {
    ($id:ident [$ety:ident]: $($elem_ty:ident),* | $($elem_name:ident),*) => {
        #[repr(simd)]
        #[derive(Copy, Clone, Debug, PartialEq)]
        pub(crate) struct $id($(pub $elem_ty),*);

        #[allow(clippy::use_self)]
        impl $id {
            #[inline]
            pub(crate) const fn new($($elem_name: $elem_ty),*) -> Self {
                $id($($elem_name),*)
            }
            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) const fn splat(value: $ety) -> Self {
                $id($({
                    #[allow(non_camel_case_types, dead_code)]
                    struct $elem_name;
                    value
                }),*)
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn extract(self, index: usize) -> $ety {
                unsafe {
                    crate::core_arch::simd_llvm::simd_extract(self, index as u32)
                }
            }
        }
    }
}

macro_rules! simd_m_ty {
    ($id:ident [$ety:ident]: $($elem_ty:ident),* | $($elem_name:ident),*) => {
        #[repr(simd)]
        #[derive(Copy, Clone, Debug, PartialEq)]
        pub(crate) struct $id($(pub $elem_ty),*);

        #[allow(clippy::use_self)]
        impl $id {
            #[inline]
            const fn bool_to_internal(x: bool) -> $ety {
                [0 as $ety, !(0 as $ety)][x as usize]
            }

            #[inline]
            pub(crate) const fn new($($elem_name: bool),*) -> Self {
                $id($(Self::bool_to_internal($elem_name)),*)
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) const fn splat(value: bool) -> Self {
                $id($({
                    #[allow(non_camel_case_types, dead_code)]
                    struct $elem_name;
                    Self::bool_to_internal(value)
                }),*)
            }

            // FIXME: Workaround rust@60637
            #[inline(always)]
            pub(crate) fn extract(self, index: usize) -> bool {
                let r: $ety = unsafe {
                    crate::core_arch::simd_llvm::simd_extract(self, index as u32)
                };
                r != 0
            }
        }
    }
}

// 16-bit wide types:

simd_ty!(u8x2[u8]: u8, u8 | x0, x1);
simd_ty!(i8x2[i8]: i8, i8 | x0, x1);

// 32-bit wide types:

simd_ty!(u8x4[u8]: u8, u8, u8, u8 | x0, x1, x2, x3);
simd_ty!(u16x2[u16]: u16, u16 | x0, x1);

simd_ty!(i8x4[i8]: i8, i8, i8, i8 | x0, x1, x2, x3);
simd_ty!(i16x2[i16]: i16, i16 | x0, x1);

// 64-bit wide types:

simd_ty!(u8x8[u8]:
         u8, u8, u8, u8, u8, u8, u8, u8
         | x0, x1, x2, x3, x4, x5, x6, x7);
simd_ty!(u16x4[u16]: u16, u16, u16, u16 | x0, x1, x2, x3);
simd_ty!(u32x2[u32]: u32, u32 | x0, x1);
simd_ty!(u64x1[u64]: u64 | x1);

simd_ty!(i8x8[i8]:
         i8, i8, i8, i8, i8, i8, i8, i8
         | x0, x1, x2, x3, x4, x5, x6, x7);
simd_ty!(i16x4[i16]: i16, i16, i16, i16 | x0, x1, x2, x3);
simd_ty!(i32x2[i32]: i32, i32 | x0, x1);
simd_ty!(i64x1[i64]: i64 | x1);

simd_ty!(f32x2[f32]: f32, f32 | x0, x1);

// 128-bit wide types:

simd_ty!(u8x16[u8]:
         u8, u8, u8, u8, u8, u8, u8, u8,
         u8, u8, u8, u8, u8, u8, u8, u8
         | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
);
simd_ty!(u16x8[u16]:
         u16, u16, u16, u16, u16, u16, u16, u16
         | x0, x1, x2, x3, x4, x5, x6, x7);
simd_ty!(u32x4[u32]: u32, u32, u32, u32 | x0, x1, x2, x3);
simd_ty!(u64x2[u64]: u64, u64 | x0, x1);

simd_ty!(i8x16[i8]:
         i8, i8, i8, i8, i8, i8, i8, i8,
         i8, i8, i8, i8, i8, i8, i8, i8
         | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
);
simd_ty!(i16x8[i16]:
         i16, i16, i16, i16, i16, i16, i16, i16
         | x0, x1, x2, x3, x4, x5, x6, x7);
simd_ty!(i32x4[i32]: i32, i32, i32, i32 | x0, x1, x2, x3);
simd_ty!(i64x2[i64]: i64, i64 | x0, x1);

simd_ty!(f32x4[f32]: f32, f32, f32, f32 | x0, x1, x2, x3);
simd_ty!(f64x2[f64]: f64, f64 | x0, x1);

simd_m_ty!(m8x16[i8]:
           i8, i8, i8, i8, i8, i8, i8, i8,
           i8, i8, i8, i8, i8, i8, i8, i8
           | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
);
simd_m_ty!(m16x8[i16]:
           i16, i16, i16, i16, i16, i16, i16, i16
           | x0, x1, x2, x3, x4, x5, x6, x7);
simd_m_ty!(m32x4[i32]: i32, i32, i32, i32 | x0, x1, x2, x3);
simd_m_ty!(m64x2[i64]: i64, i64 | x0, x1);

// 256-bit wide types:

simd_ty!(u8x32[u8]:
         u8, u8, u8, u8, u8, u8, u8, u8,
         u8, u8, u8, u8, u8, u8, u8, u8,
         u8, u8, u8, u8, u8, u8, u8, u8,
         u8, u8, u8, u8, u8, u8, u8, u8
         | x0, x1, x2, x3, x4, x5, x6, x7,
         x8, x9, x10, x11, x12, x13, x14, x15,
         x16, x17, x18, x19, x20, x21, x22, x23,
         x24, x25, x26, x27, x28, x29, x30, x31
);
simd_ty!(u16x16[u16]:
         u16, u16, u16, u16, u16, u16, u16, u16,
         u16, u16, u16, u16, u16, u16, u16, u16
         | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
);
simd_ty!(u32x8[u32]:
         u32, u32, u32, u32, u32, u32, u32, u32
         | x0, x1, x2, x3, x4, x5, x6, x7);
simd_ty!(u64x4[u64]: u64, u64, u64, u64 | x0, x1, x2, x3);

simd_ty!(i8x32[i8]:
         i8, i8, i8, i8, i8, i8, i8, i8,
         i8, i8, i8, i8, i8, i8, i8, i8,
         i8, i8, i8, i8, i8, i8, i8, i8,
         i8, i8, i8, i8, i8, i8, i8, i8
         | x0, x1, x2, x3, x4, x5, x6, x7,
         x8, x9, x10, x11, x12, x13, x14, x15,
         x16, x17, x18, x19, x20, x21, x22, x23,
         x24, x25, x26, x27, x28, x29, x30, x31
);
simd_ty!(i16x16[i16]:
         i16, i16, i16, i16, i16, i16, i16, i16,
         i16, i16, i16, i16, i16, i16, i16, i16
         | x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
);
simd_ty!(i32x8[i32]:
         i32, i32, i32, i32, i32, i32, i32, i32
         | x0, x1, x2, x3, x4, x5, x6, x7);
simd_ty!(i64x4[i64]: i64, i64, i64, i64 | x0, x1, x2, x3);

simd_ty!(f32x8[f32]:
         f32, f32, f32, f32, f32, f32, f32, f32 |
         x0, x1, x2, x3, x4, x5, x6, x7);

// 512-bit wide types:

simd_ty!(i32x16[i32]:
         i32, i32, i32, i32, i32, i32, i32, i32,
         i32, i32, i32, i32, i32, i32, i32, i32
         | x0, x1, x2, x3, x4, x5, x6, x7,
         x8, x9, x10, x11, x12, x13, x14, x15);

simd_ty!(i64x8[i64]:
         i64, i64, i64, i64, i64, i64, i64, i64
         | x0, x1, x2, x3, x4, x5, x6, x7);
