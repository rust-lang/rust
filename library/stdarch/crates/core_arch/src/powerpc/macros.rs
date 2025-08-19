macro_rules! test_impl {
    ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr:ident]) => {
        #[inline]
        #[target_feature(enable = "altivec")]
        #[cfg_attr(test, assert_instr($instr))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            $call ($($v),*)
        }
    };
    ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr_altivec:ident / $instr_vsx:ident]) => {
        test_impl! { $fun ($($v : $ty),*) -> $r [$call, $instr_altivec / $instr_vsx / $instr_vsx] }
    };
    ($fun:ident ($($v:ident : $ty:ty),*) -> $r:ty [$call:ident, $instr_altivec:ident / $instr_vsx:ident / $instr_pwr9:ident]) => {
        #[inline]
        #[target_feature(enable = "altivec")]
        #[cfg_attr(all(test, not(target_feature="vsx"), not(target_feature = "power9-vector")), assert_instr($instr_altivec))]
        #[cfg_attr(all(test, target_feature="vsx", not(target_feature = "power9-vector")), assert_instr($instr_vsx))]
        #[cfg_attr(all(test, not(target_feature="vsx"), target_feature = "power9-vector"), assert_instr($instr_pwr9))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            $call ($($v),*)
        }
    }
}

#[allow(unknown_lints, unused_macro_rules)]
macro_rules! impl_vec_trait {
    ([$Trait:ident $m:ident] $fun:ident ($a:ty)) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl $Trait for $a {
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $m(self) -> Self {
                $fun(transmute(self))
            }
        }
    };
    ([$Trait:ident $m:ident] $fun:ident ($a:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl $Trait for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $m(self) -> Self::Result {
                $fun(transmute(self))
            }
        }
    };
    ([$Trait:ident $m:ident]+ $fun:ident ($a:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl $Trait for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "altivec")]
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
        impl_vec_trait!{ [$Trait $m] $sf (vector_float) -> vector_float }
    };
    ([$Trait:ident $m:ident] $fun:ident ($a:ty, $b:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl $Trait<$b> for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $m(self, b: $b) -> Self::Result {
                $fun(transmute(self), transmute(b))
            }
        }
    };
    ([$Trait:ident $m:ident]+ $fun:ident ($a:ty, $b:ty) -> $r:ty) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        impl $Trait<$b> for $a {
            type Result = $r;
            #[inline]
            #[target_feature(enable = "altivec")]
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
    ([$Trait:ident $m:ident] ~($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident)) => {
        impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, ~vector_bool_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, ~vector_bool_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, ~vector_bool_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, ~vector_bool_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, ~vector_bool_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, ~vector_bool_int) -> vector_signed_int }
    };
    ([$Trait:ident $m:ident] ~($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m] ~($fn, $fn, $fn, $fn, $fn, $fn) }
    };
    ([$Trait:ident $m:ident] 2 ($ub:ident, $sb:ident, $uh:ident, $sh:ident, $uw:ident, $sw:ident)) => {
        impl_vec_trait!{ [$Trait $m] $ub (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m] $sb (vector_signed_char, vector_signed_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m] $uh (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m] $sh (vector_signed_short, vector_signed_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m] $uw (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m] $sw (vector_signed_int, vector_signed_int) -> vector_signed_int }
    };
    ([$Trait:ident $m:ident] 2 ($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m] ($fn, $fn, $fn, $fn, $fn, $fn) }
    };
    ([$Trait:ident $m:ident]+ 2b ($b:ident, $h:ident, $w:ident)) => {
        impl_vec_trait!{ [$Trait $m]+ $b (vector_bool_char, vector_bool_char) -> vector_bool_char }
        impl_vec_trait!{ [$Trait $m]+ $b (vector_unsigned_char, vector_unsigned_char) -> vector_unsigned_char }
        impl_vec_trait!{ [$Trait $m]+ $b (vector_signed_char, vector_signed_char) -> vector_signed_char }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_bool_short, vector_bool_short) -> vector_bool_short }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_unsigned_short, vector_unsigned_short) -> vector_unsigned_short }
        impl_vec_trait!{ [$Trait $m]+ $h (vector_signed_short, vector_signed_short) -> vector_signed_short }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_bool_int, vector_bool_int) -> vector_bool_int }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_unsigned_int, vector_unsigned_int) -> vector_unsigned_int }
        impl_vec_trait!{ [$Trait $m]+ $w (vector_signed_int, vector_signed_int) -> vector_signed_int }
    };
    ([$Trait:ident $m:ident]+ 2b ($fn:ident)) => {
        impl_vec_trait!{ [$Trait $m]+ 2b ($fn, $fn, $fn) }
    };
}

macro_rules! s_t_l {
    (i32x4) => {
        vector_signed_int
    };
    (i16x8) => {
        vector_signed_short
    };
    (i8x16) => {
        vector_signed_char
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
}

macro_rules! t_t_l {
    (i32) => {
        vector_signed_int
    };
    (i16) => {
        vector_signed_short
    };
    (i8) => {
        vector_signed_char
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
}

macro_rules! t_t_s {
    (i32) => {
        i32x4
    };
    (i16) => {
        i16x8
    };
    (i8) => {
        i8x16
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
    (vector_unsigned_char) => {
        vector_unsigned_char
    };
    (vector_unsigned_short) => {
        vector_unsigned_short
    };
    (vector_unsigned_int) => {
        vector_unsigned_int
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
    (vector_float) => {
        vector_unsigned_int
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
    (vector_signed_char) => {
        vector_bool_char
    };
    (vector_signed_short) => {
        vector_bool_short
    };
    (vector_signed_int) => {
        vector_bool_int
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
    (vector_float) => {
        vector_bool_int
    };
}

macro_rules! impl_from {
    ($s: ident) => {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
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
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
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
pub(crate) use s_t_l;
pub(crate) use t_b;
pub(crate) use t_t_l;
pub(crate) use t_t_s;
pub(crate) use t_u;
pub(crate) use test_impl;
