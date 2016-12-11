macro_rules! define_ty {
    ($name:ident, $($elty:ident),+) => {
        #[repr(simd)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[allow(non_camel_case_types)]
        pub struct $name($($elty),*);
    }
}

macro_rules! define_impl {
    ($name:ident, $elemty:ident, $nelems:expr,
     $($elname:ident),+) => {
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

            #[inline(always)]
            pub fn extract(self, idx: u32) -> $elemty {
                assert!(idx < $nelems);
                unsafe { simd_extract(self, idx) }
            }

            #[inline(always)]
            pub fn insert(self, idx: u32, val: $elemty) -> $name {
                assert!(idx < $nelems);
                unsafe { simd_insert(self, idx, val) }
            }
        }
    }
}

macro_rules! define_from {
    ($to:ident, $($from:ident),+) => {
        $(
            impl From<$from> for $to {
                fn from(f: $from) -> $to {
                    unsafe { ::std::mem::transmute(f) }
                }
            }
        )+
    }
}

macro_rules! define_common_ops {
    ($($ty:ident),+) => {
        $(
            impl ::std::ops::Add for $ty {
                type Output = Self;
                #[inline(always)]
                fn add(self, other: Self) -> Self {
                    unsafe { simd_add(self, other) }
                }
            }

            impl ::std::ops::Sub for $ty {
                type Output = Self;
                #[inline(always)]
                fn sub(self, other: Self) -> Self {
                    unsafe { simd_sub(self, other) }
                }
            }

            impl ::std::ops::Mul for $ty {
                type Output = Self;
                #[inline(always)]
                fn mul(self, other: Self) -> Self {
                    unsafe { simd_mul(self, other) }
                }
            }
        )+
    }
}

macro_rules! define_float_ops {
    ($($ty:ident),+) => {
        $(
            impl ::std::ops::Div for $ty {
                type Output = Self;
                #[inline(always)]
                fn div(self, other: Self) -> Self {
                    unsafe { simd_div(self, other) }
                }
            }
        )+
    }
}

macro_rules! define_shifts {
    ($ty:ident, $elem:ident, $($by:ident),+) => {
        $(
            impl ::std::ops::Shl<$by> for $ty {
                type Output = Self;
                #[inline(always)]
                fn shl(self, other: $by) -> Self {
                    unsafe { simd_shl(self, $ty::splat(other as $elem)) }
                }
            }
            impl ::std::ops::Shr<$by> for $ty {
                type Output = Self;
                #[inline(always)]
                fn shr(self, other: $by) -> Self {
                    unsafe { simd_shr(self, $ty::splat(other as $elem)) }
                }
            }
        )+
    }
}

macro_rules! define_integer_ops {
    ($(($ty:ident, $elem:ident)),+) => {
        $(
            impl ::std::ops::BitAnd for $ty {
                type Output = Self;
                #[inline(always)]
                fn bitand(self, other: Self) -> Self {
                    unsafe { simd_and(self, other) }
                }
            }
            impl ::std::ops::BitOr for $ty {
                type Output = Self;
                #[inline(always)]
                fn bitor(self, other: Self) -> Self {
                    unsafe { simd_or(self, other) }
                }
            }
            impl ::std::ops::BitXor for $ty {
                type Output = Self;
                #[inline(always)]
                fn bitxor(self, other: Self) -> Self {
                    unsafe { simd_xor(self, other) }
                }
            }
            impl ::std::ops::Not for $ty {
                type Output = Self;
                #[inline(always)]
                fn not(self) -> Self {
                    $ty::splat(!0) ^ self
                }
            }
            define_shifts!(
                $ty, $elem,
                u8, u16, u32, u64, usize,
                i8, i16, i32, i64, isize);
        )+
    }
}
