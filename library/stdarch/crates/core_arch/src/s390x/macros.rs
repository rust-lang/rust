#![allow(unused_macros)] // FIXME remove when more tests are added
#![allow(unused_imports)] // FIXME remove when more tests are added

macro_rules! test_impl {
    ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, _]) => {
        #[inline]
        #[target_feature(enable = "vector")]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            $call ($($v),*)
        }
    };
    ($fun:ident +($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr:ident]) => {
        #[inline]
        #[target_feature(enable = "vector")]
        #[cfg_attr(test, assert_instr($instr))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            transmute($call ($($v),*))
        }
    };
    ($fun:ident +($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $tf:literal $instr:ident]) => {
        #[inline]
        #[target_feature(enable = "vector")]
        #[cfg_attr(all(test, target_feature = $tf), assert_instr($instr))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            transmute($call ($($v),*))
        }
    };
    ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $tf:literal $instr:ident]) => {
        #[inline]
        #[target_feature(enable = "vector")]
        #[cfg_attr(all(test, target_feature = $tf), assert_instr($instr))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            $call ($($v),*)
        }
    };
    ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr:ident]) => {
        #[inline]
        #[target_feature(enable = "vector")]
        #[cfg_attr(test, assert_instr($instr))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            $call ($($v),*)
        }
    };
}

#[allow(unknown_lints, unused_macro_rules)]
macro_rules! impl_vec_trait {
    ([$Trait:ident $m:ident] $fun:ident ($a:ty)) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl $Trait for $a {
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $m(self) -> Self {
                $fun(transmute(self))
            }
        }
    };
    ([$Trait:ident $m:ident]+ $fun:ident ($a:ty)) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl $Trait for $a {
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $m(self) -> Self {
                transmute($fun(transmute(self)))
            }
        }
    };
    ([$Trait:ident $m:ident] $fun:ident ($a:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl $Trait for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $m(self) -> Self::Result {
                $fun(transmute(self))
            }
        }
    };
    ([$Trait:ident $m:ident]+ $fun:ident ($a:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl $Trait for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $m(self) -> Self::Result {
                transmute($fun(transmute(self)))
            }
        }
    };
    ([$Trait:ident $m:ident] 1 ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident, $sf: ident)) => {
        impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int) -> vector_signed_int }
        impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_long_long) -> vector_unsigned_long_long }
        impl_vec_trait!{ [$Trait $m] $sw (vector_signed_long_long) -> vector_signed_long_long }
        impl_vec_trait!{ [$Trait $m] $sf (vector_float) -> vector_float }
        impl_vec_trait!{ [$Trait $m] $sf (vector_double) -> vector_double }
    };
    ([$Trait:ident $m:ident] $fun:ident ($a:ty, $b:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl $Trait<$b> for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $m(self, b: $b) -> Self::Result {
                $fun(transmute(self), transmute(b))
            }
        }
    };
    ([$Trait:ident $m:ident]+ $fun:ident ($a:ty, $b:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl $Trait<$b> for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "vector")]
            unsafe fn $m(self, b: $b) -> Self::Result {
                transmute($fun(transmute(self), transmute(b)))
            }
        }
    };
    ([$Trait:ident $m:ident] $fun:ident ($a:ty, ~$b:ty) -> $r:ty) => {
        impl_vec_trait!{ [$Trait $m] $fun ($a, $a) -> $r }
        impl_vec_trait!{ [$Trait $m] $fun ($a, $b) -> $r }
        impl_vec_trait!{ [$Trait $m] $fun ($b, $a) -> $r }
    };
    ([$Trait:ident $m:ident] ~($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident, $ug:ident, $sg:ident)) => {
        impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, ~vector_bool_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, ~vector_bool_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, ~vector_bool_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, ~vector_bool_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, ~vector_bool_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, ~vector_bool_int) -> vector_signed_int }
        impl_vec_trait!{ [$Trait $m] $ug (vector_unsigned_long_long, ~vector_bool_long_long) -> vector_unsigned_long_long }
        impl_vec_trait!{ [$Trait $m] $sg (vector_signed_long_long, ~vector_bool_long_long) -> vector_signed_long_long }
    };
    ([$Trait:ident $m:ident] ~($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m] ~($fn, $fn, $fn, $fn, $fn, $fn, $fn, $fn) }
    };
    ([$Trait:ident $m:ident] 2 ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident, $ug:ident, $sg:ident)) => {
        impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, vector_signed_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, vector_signed_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, vector_signed_int) -> vector_signed_int }
        impl_vec_trait!{ [$Trait $m] $ug (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
        impl_vec_trait!{ [$Trait $m] $sg (vector_signed_long_long, vector_signed_long_long) -> vector_signed_long_long }
    };
    ([$Trait:ident $m:ident] 2 ($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m] ($fn, $fn, $fn, $fn, $fn, $fn, $fn, $fn) }
    };
    ([$Trait:ident $m:ident]+ 2b ($b:ident, $h:ident, $w:ident, $g:ident)) => {
        impl_vec_trait!{ [$Trait $m]+ $b (vector_bool_char, vector_bool_char) -> vector_bool_char }
        impl_vec_trait!{ [$Trait $m]+ $b (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m]+ $b (vector_signed_char, vector_signed_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_bool_short, vector_bool_short) -> vector_bool_short }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_signed_short, vector_signed_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_bool_int, vector_bool_int) -> vector_bool_int }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_signed_int, vector_signed_int) -> vector_signed_int }
        impl_vec_trait!{ [$Trait $m]+ $g (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
        impl_vec_trait!{ [$Trait $m]+ $g (vector_signed_long_long, vector_signed_long_long) -> vector_signed_long_long }
    };
    ([$Trait:ident $m:ident]+ 2b ($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m]+ 2b ($fn, $fn, $fn, $fn) }
    };
    ([$Trait:ident $m:ident]+ 2c ($b:ident, $h:ident, $w:ident, $g:ident, $s:ident, $d:ident)) => {
        impl_vec_trait!{ [$Trait $m]+ $b (vector_bool_char, vector_bool_char) -> vector_bool_char }
        impl_vec_trait!{ [$Trait $m]+ $b (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m]+ $b (vector_signed_char, vector_signed_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_bool_short, vector_bool_short) -> vector_bool_short }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_signed_short, vector_signed_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_bool_int, vector_bool_int) -> vector_bool_int }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_signed_int, vector_signed_int) -> vector_signed_int }
        impl_vec_trait!{ [$Trait $m]+ $g (vector_unsigned_long_long, vector_unsigned_long_long) -> vector_unsigned_long_long }
        impl_vec_trait!{ [$Trait $m]+ $g (vector_signed_long_long, vector_signed_long_long) -> vector_signed_long_long }
        impl_vec_trait!{ [$Trait $m]+ $s (vector_float, vector_float) -> vector_float }
        impl_vec_trait!{ [$Trait $m]+ $d (vector_double, vector_double) -> vector_double }
    };
    ([$Trait:ident $m:ident]+ 2c ($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m]+ 2c ($fn, $fn, $fn, $fn, $fn, $fn) }
    };
}

macro_rules! s_t_l {
    (i64x2) => {
        vector_signed_long_long
    };
    (i32x4) => {
        vector_signed_int
    };
    (i16x8) => {
        vector_signed_short
    };
    (i8x16) => {
        vector_signed_char
    };

    (u64x2) => {
        vector_unsigned_long_long
    };
    (u32x4) => {
        vector_unsigned_int
    };
    (u16x8) => {
        vector_unsigned_short
    };
    (u8x16) => {
        vector_unsigned_char
    };

    (f32x4) => {
        vector_float
    };
    (f64x2) => {
        vector_double
    };
}

macro_rules! l_t_t {
    (vector_signed_long_long) => {
        i64
    };
    (vector_signed_int) => {
        i32
    };
    (vector_signed_short) => {
        i16
    };
    (vector_signed_char) => {
        i8
    };

    (vector_unsigned_long_long ) => {
        u64
    };
    (vector_unsigned_int ) => {
        u32
    };
    (vector_unsigned_short ) => {
        u16
    };
    (vector_unsigned_char ) => {
        u8
    };

    (vector_bool_long_long ) => {
        u64
    };
    (vector_bool_int ) => {
        u32
    };
    (vector_bool_short ) => {
        u16
    };
    (vector_bool_char ) => {
        u8
    };

    (vector_float) => {
        f32
    };
    (vector_double) => {
        f64
    };
}

macro_rules! t_t_l {
    (i64) => {
        vector_signed_long_long
    };
    (i32) => {
        vector_signed_int
    };
    (i16) => {
        vector_signed_short
    };
    (i8) => {
        vector_signed_char
    };

    (u64) => {
        vector_unsigned_long_long
    };
    (u32) => {
        vector_unsigned_int
    };
    (u16) => {
        vector_unsigned_short
    };
    (u8) => {
        vector_unsigned_char
    };

    (f32) => {
        vector_float
    };
    (f64) => {
        vector_double
    };
}

macro_rules! t_t_s {
    (i64) => {
        i64x2
    };
    (i32) => {
        i32x4
    };
    (i16) => {
        i16x8
    };
    (i8) => {
        i8x16
    };

    (u64) => {
        u64x2
    };
    (u32) => {
        u32x4
    };
    (u16) => {
        u16x8
    };
    (u8) => {
        u8x16
    };

    (f32) => {
        f32x4
    };
    (f64) => {
        f64x2
    };
}

macro_rules! t_u {
    (vector_bool_char) => {
        vector_unsigned_char
    };
    (vector_bool_short) => {
        vector_unsigned_short
    };
    (vector_bool_int) => {
        vector_unsigned_int
    };
    (vector_bool_long_long) => {
        vector_unsigned_long_long
    };
    (vector_unsigned_char) => {
        vector_unsigned_char
    };
    (vector_unsigned_short) => {
        vector_unsigned_short
    };
    (vector_unsigned_int) => {
        vector_unsigned_int
    };
    (vector_unsigned_long_long) => {
        vector_unsigned_long_long
    };
    (vector_signed_char) => {
        vector_unsigned_char
    };
    (vector_signed_short) => {
        vector_unsigned_short
    };
    (vector_signed_int) => {
        vector_unsigned_int
    };
    (vector_signed_long_long) => {
        vector_unsigned_long_long
    };
    (vector_float) => {
        vector_unsigned_int
    };
    (vector_double) => {
        vector_unsigned_long_long
    };
}

macro_rules! t_b {
    (vector_bool_char) => {
        vector_bool_char
    };
    (vector_bool_short) => {
        vector_bool_short
    };
    (vector_bool_int) => {
        vector_bool_int
    };
    (vector_bool_long_long) => {
        vector_bool_long_long
    };
    (vector_signed_char) => {
        vector_bool_char
    };
    (vector_signed_short) => {
        vector_bool_short
    };
    (vector_signed_int) => {
        vector_bool_int
    };
    (vector_signed_long_long) => {
        vector_bool_long_long
    };
    (vector_unsigned_char) => {
        vector_bool_char
    };
    (vector_unsigned_short) => {
        vector_bool_short
    };
    (vector_unsigned_int) => {
        vector_bool_int
    };
    (vector_unsigned_long_long) => {
        vector_bool_long_long
    };
    (vector_float) => {
        vector_bool_int
    };
    (vector_double) => {
        vector_bool_long_long
    };
}

macro_rules! impl_from {
    ($s: ident) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl From<$s> for s_t_l!($s) {
            #[inline]
            fn from (v: $s) -> Self {
                unsafe {
                    transmute(v)
                }
            }
        }
    };
    ($($s: ident),*) => {
        $(
            impl_from! { $s }
        )*
    };
}

macro_rules! impl_neg {
    ($s: ident : $zero: expr) => {
        #[unstable(feature = "stdarch_s390x", issue = "135681")]
        impl crate::ops::Neg for s_t_l!($s) {
            type Output = s_t_l!($s);
            #[inline]
            fn neg(self) -> Self::Output {
                unsafe { simd_neg(self) }
            }
        }
    };
}

pub(crate) use impl_from;
pub(crate) use impl_neg;
pub(crate) use impl_vec_trait;
pub(crate) use l_t_t;
pub(crate) use s_t_l;
pub(crate) use t_b;
pub(crate) use t_t_l;
pub(crate) use t_t_s;
pub(crate) use t_u;
pub(crate) use test_impl;
