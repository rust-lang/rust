macro_rules! define_ty {
    ($name:ident, $($elty:ident),+) => {
        #[repr(simd)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[allow(non_camel_case_types)]
        pub struct $name($($elty),*);
    }
}

macro_rules! define_impl {
    (
        $name:ident, $elemty:ident, $nelems:expr, $boolname:ident,
        $($elname:ident),+
    ) => {
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
            pub fn extract(self, idx: u32) -> $elemty {
                assert!(idx < $nelems);
                unsafe { simd_extract(self, idx) }
            }

            #[inline]
            pub fn replace(self, idx: u32, val: $elemty) -> $name {
                assert!(idx < $nelems);
                unsafe { simd_insert(self, idx, val) }
            }

            #[inline]
            pub fn store(self, slice: &mut [$elemty], offset: usize) {
                assert!(slice[offset..].len() >= $nelems);
                unsafe { self.store_unchecked(slice, offset) }
            }

            #[inline]
            pub unsafe fn store_unchecked(
                self,
                slice: &mut [$elemty],
                offset: usize,
            ) {
                use std::mem::size_of;
                use std::ptr;

                ptr::copy_nonoverlapping(
                    &self as *const $name as *const u8,
                    slice.get_unchecked_mut(offset) as *mut $elemty as *mut u8,
                    size_of::<$name>());
            }

            #[inline]
            pub fn load(slice: &[$elemty], offset: usize) -> $name {
                assert!(slice[offset..].len() >= $nelems);
                unsafe { $name::load_unchecked(slice, offset) }
            }

            #[inline]
            pub unsafe fn load_unchecked(
                slice: &[$elemty],
                offset: usize,
            ) -> $name {
                use std::mem::size_of;
                use std::ptr;

                let mut x = $name::splat(0 as $elemty);
                ptr::copy_nonoverlapping(
                    slice.get_unchecked(offset) as *const $elemty as *const u8,
                    &mut x as *mut $name as *mut u8,
                    size_of::<$name>());
                x
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

            impl ::std::fmt::LowerHex for $ty {
                fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                    write!(f, "{}(", stringify!($ty))?;
                    let n = ::std::mem::size_of_val(self) / ::std::mem::size_of::<$elem>();
                    for i in 0..n {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:#x}", self.extract(i as u32))?;
                    }
                    write!(f, ")")
                }
            }
        )+
    }
}

macro_rules! define_casts {
    ($(($ty:ident, $floatty:ident, $floatcast:ident)),+) => {
        $(
            impl $ty {
                #[inline]
                pub fn $floatcast(self) -> ::$floatty {
                    unsafe { simd_cast(self) }
                }
            }
        )+
    }
}
