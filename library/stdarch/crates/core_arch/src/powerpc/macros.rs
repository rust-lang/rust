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
        #[inline]
        #[target_feature(enable = "altivec")]
        #[cfg_attr(all(test, not(target_feature="vsx")), assert_instr($instr_altivec))]
        #[cfg_attr(all(test, target_feature="vsx"), assert_instr($instr_vsx))]
        pub unsafe fn $fun ($($v : $ty),*) -> $r {
            $call ($($v),*)
        }
    }
}

#[allow(unknown_lints, unused_macro_rules)]
macro_rules! impl_vec_trait {
    ([$Trait:ident $m:ident] $fun:ident ($a:ty)) => {
        impl $Trait for $a {
            #[inline]
            #[target_feature(enable = "altivec")]
            unsafe fn $m(self) -> Self {
                $fun(transmute(self))
            }
        }
    };
    ([$Trait:ident $m:ident] $fun:ident ($a:ty) -> $r:ty) => {
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
    }
}
