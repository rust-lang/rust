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
        impl From<__m64> for $name {
            #[inline]
            fn from(v: __m64) -> $name { unsafe { transmute(v) } }
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
            pub fn as_m64(self) -> __m64 { unsafe { transmute(self) } }
            #[inline]
            pub fn as_u64(self) -> u64 { unsafe { transmute(self) } }
        }
    }
}

define_ty! { __m64, u64 }

define_ty_internal! { boolu64x1, u64 }
define_ty_internal! { boolu32x2, u32, u32 }

define_ty_internal! { u64x1, u64 }
define_impl! { u64x1, boolu64x1, u64, 1, x0 }

define_ty_internal! { u32x2, u32, u32 }
define_impl! { u32x2, boolu32x2, u32, 2, x0, x1 }
