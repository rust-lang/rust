use std::mem::transmute;

use simd::*;

macro_rules! define_ty {
    ($name:ident, $($elty:ident),+) => {
        #[repr(simd)]
        #[derive(Clone, Copy, Debug)]
        #[allow(non_camel_case_types)]
        pub struct $name($($elty),*);
    }
}

macro_rules! define_ty_internal {
    ($name:ident, $($elty:ident),+) => {
        #[repr(simd)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[allow(non_camel_case_types)]
        pub struct $name($($elty),*);
    }
}

macro_rules! define_impl {
    ($name:ident, $boolname:ident, $elemty:ident, $nelems:expr,
     $($elname:ident),+) => {
        impl From<__m128> for $name {
            #[inline]
            fn from(v: __m128) -> $name { unsafe { transmute(v) } }
        }

        impl From<__m128i> for $name {
            #[inline]
            fn from(v: __m128i) -> $name { unsafe { transmute(v) } }
        }

        impl From<__m128d> for $name {
            #[inline]
            fn from(v: __m128d) -> $name { unsafe { transmute(v) } }
        }

        impl $name {
            #[inline]
            pub fn new($($elname: $elemty),*) -> $name {
                $name($($elname),*)
            }

            #[inline]
            pub fn splat(value: $elemty) -> $name {
                $name($({
                    #[allow(non_camel_case_types, dead_code)]
                    struct $elname;
                    value
                }),*)
            }

            #[inline]
            pub fn eq(self, other: $name) -> $boolname {
                unsafe { simd_eq(self, other) }
            }

            #[inline]
            pub fn ne(self, other: $name) -> $boolname {
                unsafe { simd_ne(self, other) }
            }

            #[inline]
            pub fn lt(self, other: $name) -> $boolname {
                unsafe { simd_lt(self, other) }
            }

            #[inline]
            pub fn le(self, other: $name) -> $boolname {
                unsafe { simd_le(self, other) }
            }

            #[inline]
            pub fn gt(self, other: $name) -> $boolname {
                unsafe { simd_gt(self, other) }
            }

            #[inline]
            pub fn ge(self, other: $name) -> $boolname {
                unsafe { simd_ge(self, other) }
            }

            #[inline]
            pub unsafe fn extract(self, idx: u32) -> $elemty {
                debug_assert!(idx < $nelems);
                simd_extract(self, idx)
            }

            #[inline]
            pub unsafe fn insert(self, idx: u32, val: $elemty) -> $name {
                debug_assert!(idx < $nelems);
                simd_insert(self, idx, val)
            }

            #[inline]
            pub fn as_m128(self) -> __m128 { unsafe { transmute(self) } }
            #[inline]
            pub fn as_m128d(self) -> __m128d { unsafe { transmute(self) } }
            #[inline]
            pub fn as_m128i(self) -> __m128i { unsafe { transmute(self) } }
            #[inline]
            pub fn as_f32x4(self) -> f32x4 { unsafe { transmute(self) } }
            #[inline]
            pub fn as_f64x2(self) -> f64x2 { unsafe { transmute(self) } }
            #[inline]
            pub fn as_u8x16(self) -> u8x16 { unsafe { transmute(self) } }
        }
    }
}

define_ty! { __m128, f32, f32, f32, f32 }
define_ty! { __m128d, f64, f64 }
define_ty! { __m128i, u64, u64 }

define_ty_internal! { boolu64x2, u64, u64 }
define_ty_internal! { boolu32x4, u32, u32, u32, u32 }
define_ty_internal! { boolu16x8, u16, u16, u16, u16, u16, u16, u16, u16 }
define_ty_internal! {
    boolu8x16, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8
}

define_ty_internal! { f64x2, f64, f64 }
define_impl! { f64x2, boolu64x2, f64, 2, x0, x1 }

define_ty_internal! { f32x4, f32, f32, f32, f32 }
define_impl! { f32x4, boolu32x4, f32, 2, x0, x1, x2, x3 }

define_ty_internal! { u64x2, u64, u64 }
define_impl! { u64x2, boolu64x2, u64, 2, x0, x1 }

define_ty_internal! { i64x2, i64, i64 }
define_impl! { i64x2, boolu64x2, i64, 2, x0, x1 }

define_ty_internal! { u32x4, u32, u32, u32, u32 }
define_impl! { u32x4, boolu32x4, u32, 4, x0, x1, x2, x3 }

define_ty_internal! { i32x4, i32, i32, i32, i32 }
define_impl! { i32x4, boolu32x4, i32, 4, x0, x1, x2, x3 }

define_ty_internal! { u16x8, u16, u16, u16, u16, u16, u16, u16, u16 }
define_impl! { u16x8, boolu16x8, u16, 8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty_internal! { i16x8, i16, i16, i16, i16, i16, i16, i16, i16 }
define_impl! { i16x8, boolu16x8, i16, 8, x0, x1, x2, x3, x4, x5, x6, x7 }

define_ty_internal! {
    u8x16, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8
}
define_impl! {
    u8x16, boolu8x16, u8, 16,
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
}

define_ty_internal! {
    i8x16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
}
define_impl! {
    i8x16, boolu8x16, i8, 16,
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
}
