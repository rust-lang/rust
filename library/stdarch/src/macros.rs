//! Utility macros

macro_rules! define_ty {
    ($name:ident, $($elty:ident),+) => {
        #[repr(simd)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[allow(non_camel_case_types)]
        pub struct $name($($elty),*);
    }
}

macro_rules! define_ty_doc {
    ($name:ident, $($elty:ident),+ | $(#[$doc:meta])*) => {
        $(#[$doc])*
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
            #[inline(always)]
            pub const fn new($($elname: $elemty),*) -> $name {
                $name($($elname),*)
            }

            #[inline(always)]
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
            pub fn replace(self, idx: u32, val: $elemty) -> $name {
                assert!(idx < $nelems);
                unsafe { simd_insert(self, idx, val) }
            }

            #[inline(always)]
            pub fn store(self, slice: &mut [$elemty], offset: usize) {
                assert!(slice[offset..].len() >= $nelems);
                unsafe { self.store_unchecked(slice, offset) }
            }

            #[inline(always)]
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

            #[inline(always)]
            pub fn load(slice: &[$elemty], offset: usize) -> $name {
                assert!(slice[offset..].len() >= $nelems);
                unsafe { $name::load_unchecked(slice, offset) }
            }

            #[inline(always)]
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

            #[inline(always)]
            pub fn eq(self, other: $name) -> $boolname {
                unsafe { simd_eq(self, other) }
            }

            #[inline(always)]
            pub fn ne(self, other: $name) -> $boolname {
                unsafe { simd_ne(self, other) }
            }

            #[inline(always)]
            pub fn lt(self, other: $name) -> $boolname {
                unsafe { simd_lt(self, other) }
            }

            #[inline(always)]
            pub fn le(self, other: $name) -> $boolname {
                unsafe { simd_le(self, other) }
            }

            #[inline(always)]
            pub fn gt(self, other: $name) -> $boolname {
                unsafe { simd_gt(self, other) }
            }

            #[inline(always)]
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
                #[inline(always)]
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

            impl ::std::ops::Div for $ty {
                type Output = Self;
                #[inline(always)]
                fn div(self, other: Self) -> Self {
                    unsafe { simd_div(self, other) }
                }
            }

            impl ::std::ops::Rem for $ty {
                type Output = Self;
                #[inline(always)]
                fn rem(self, other: Self) -> Self {
                    unsafe { simd_rem(self, other) }
                }
            }

            impl ::std::ops::AddAssign for $ty {
                #[inline(always)]
                fn add_assign(&mut self, other: Self) {
                    *self = *self + other;
                }
            }

            impl ::std::ops::SubAssign for $ty {
                #[inline(always)]
                fn sub_assign(&mut self, other: Self) {
                    *self = *self - other;
                }
            }

            impl ::std::ops::MulAssign for $ty {
                #[inline(always)]
                fn mul_assign(&mut self, other: Self) {
                    *self = *self * other;
                }
            }

            impl ::std::ops::DivAssign for $ty {
                #[inline(always)]
                fn div_assign(&mut self, other: Self) {
                    *self = *self / other;
                }
            }

            impl ::std::ops::RemAssign for $ty {
                #[inline(always)]
                fn rem_assign(&mut self, other: Self) {
                    *self = *self % other;
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

            impl ::std::ops::ShlAssign<$by> for $ty {
                #[inline(always)]
                fn shl_assign(&mut self, other: $by) {
                    *self = *self << other;
                }
            }
            impl ::std::ops::ShrAssign<$by> for $ty {
                #[inline(always)]
                fn shr_assign(&mut self, other: $by) {
                    *self = *self >> other;
                }
            }

        )+
    }
}

macro_rules! define_float_ops {
    ($($ty:ident),+) => {
        $(
            impl ::std::ops::Neg for $ty {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    Self::splat(-1.0) * self
                }
            }
        )+
    };
}

macro_rules! define_signed_integer_ops {
    ($($ty:ident),+) => {
        $(
            impl ::std::ops::Neg for $ty {
                type Output = Self;
                #[inline(always)]
                fn neg(self) -> Self {
                    Self::splat(-1) * self
                }
            }
        )+
    };
}

macro_rules! define_integer_ops {
    ($(($ty:ident, $elem:ident)),+) => {
        $(
            impl ::std::ops::Not for $ty {
                type Output = Self;
                #[inline(always)]
                fn not(self) -> Self {
                    $ty::splat(!0) ^ self
                }
            }

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
            impl ::std::ops::BitAndAssign for $ty {
                #[inline(always)]
                fn bitand_assign(&mut self, other: Self) {
                    *self = *self & other;
                }
            }
            impl ::std::ops::BitOrAssign for $ty {
                #[inline(always)]
                fn bitor_assign(&mut self, other: Self) {
                    *self = *self | other;
                }
            }
            impl ::std::ops::BitXorAssign for $ty {
                #[inline(always)]
                fn bitxor_assign(&mut self, other: Self) {
                    *self = *self ^ other;
                }
            }

            define_shifts!(
                $ty, $elem,
                u8, u16, u32, u64, usize,
                i8, i16, i32, i64, isize);

            impl ::std::fmt::LowerHex for $ty {
                fn fmt(&self, f: &mut ::std::fmt::Formatter)
                       -> ::std::fmt::Result {
                    write!(f, "{}(", stringify!($ty))?;
                    let n = ::std::mem::size_of_val(self)
                        / ::std::mem::size_of::<$elem>();
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
    ($(($fromty:ident, $toty:ident, $cast:ident)),+) => {
        $(
            impl $fromty {
                #[inline(always)]
                pub fn $cast(self) -> ::simd::$toty {
                    unsafe { simd_cast(self) }
                }
            }
        )+
    }
}

/// Is a feature supported by the host CPU?
///
/// This macro performs run-time feature detection. It returns true if the host
/// CPU in which the binary is running on supports a particular feature.
#[macro_export]
macro_rules! cfg_feature_enabled {
    ($name:tt) => (
        {
            #[cfg(target_feature = $name)]
            {
                true
            }
            #[cfg(not(target_feature = $name))]
            {
                __unstable_detect_feature!($name)
            }
        }
    )
}

/// On ARM features are only detected at compile-time using
/// cfg(target_feature), so if this macro is executed the
/// feature is not supported.
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("neon") => { false };
    ($t:tt) => { compile_error!(concat!("unknown target feature: ", $t)) };
}

/// In all unsupported architectures using the macro is an error
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64",
              target_arch = "arm", target_arch = "aarch64")))]
#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ($t:tt) => { compile_error!(concat!("unknown target feature: ", $t)) };
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_macros() {
        assert!(cfg_feature_enabled!("sse"));
    }
}


#[cfg(test)]
#[macro_export]
macro_rules! test_arithmetic_ {
        ($tn:ident, $zero:expr, $one:expr, $two:expr, $four:expr) => {
            {
                 let z = $tn::splat($zero);
                 let o = $tn::splat($one);
                 let t = $tn::splat($two);
                 let f = $tn::splat($four);

                 // add
                 assert_eq!(z + z, z);
                 assert_eq!(o + z, o);
                 assert_eq!(t + z, t);
                 assert_eq!(t + t, f);
                 // sub
                 assert_eq!(z - z, z);
                 assert_eq!(o - z, o);
                 assert_eq!(t - z, t);
                 assert_eq!(f - t, t);
                 assert_eq!(f - o - o, t);
                 // mul
                 assert_eq!(z * z, z);
                 assert_eq!(z * o, z);
                 assert_eq!(z * t, z);
                 assert_eq!(o * t, t);
                 assert_eq!(t * t, f);
                 // div
                 assert_eq!(z / o, z);
                 assert_eq!(t / o, t);
                 assert_eq!(f / o, f);
                 assert_eq!(t / t, o);
                 assert_eq!(f / t, t);
                 // rem
                 assert_eq!(o % o, z);
                 assert_eq!(f % t, z);

                {
                    let mut v = z;
                    assert_eq!(v, z);
                    v += o;  // add_assign
                    assert_eq!(v, o);
                    v -= o; // sub_assign
                    assert_eq!(v, z);
                    v = t;
                    v *= o; // mul_assign
                    assert_eq!(v, t);
                    v *= t;
                    assert_eq!(v, f);
                    v /= o; // div_assign
                    assert_eq!(v, f);
                    v /= t;
                    assert_eq!(v, t);
                    v %= t; // rem_assign
                    assert_eq!(v, z);
                }
            }
        };
    }

#[cfg(test)]
#[macro_export]
    macro_rules! test_neg_ {
        ($tn:ident, $zero:expr, $one:expr, $two:expr, $four:expr) => {
            {
                let z = $tn::splat($zero);
                let o = $tn::splat($one);
                let t = $tn::splat($two);
                let f = $tn::splat($four);

                let nz = $tn::splat(-$zero);
                let no = $tn::splat(-$one);
                let nt = $tn::splat(-$two);
                let nf = $tn::splat(-$four);

                assert_eq!(-z, nz);
                assert_eq!(-o, no);
                assert_eq!(-t, nt);
                assert_eq!(-f, nf);
            }
        };
    }

#[cfg(test)]
#[macro_export]
macro_rules! test_bit_arithmetic_ {
    ($tn:ident) => {
        {
            let z = $tn::splat(0);
            let o = $tn::splat(1);
            let t = $tn::splat(2);
            let f = $tn::splat(4);
            let m = $tn::splat(!z.extract(0));

            // shr
            assert_eq!(o >> 1, z);
            assert_eq!(t >> 1, o);
            assert_eq!(f >> 1, t);
            // shl
            assert_eq!(o << 1, t);
            assert_eq!(o << 2, f);
            assert_eq!(t << 1, f);
            // bitand
            assert_eq!(o & o, o);
            assert_eq!(t & t, t);
            assert_eq!(t & o, z);
            // bitor
            assert_eq!(o | o, o);
            assert_eq!(t | t, t);
            assert_eq!(z | o, o);
            // bitxor
            assert_eq!(o ^ o, z);
            assert_eq!(t ^ t, z);
            assert_eq!(z ^ o, o);
            // not
            assert_eq!(!z, m);
            assert_eq!(!m, z);

            {  // shr_assign
                let mut v = o;
                v >>= 1;
                assert_eq!(v, z);
            }
            {  // shl_assign
                let mut v = o;
                v <<= 1;
                assert_eq!(v, t);
            }
            {  // and_assign
                let mut v = o;
                v &= t;
                assert_eq!(v, z);
            }
            {  // or_assign
                let mut v = z;
                v |= o;
                assert_eq!(v, o);
            }
            {  // xor_assign
                let mut v = z;
                v ^= o;
                assert_eq!(v, o);
            }
        }
    };
}


#[cfg(test)]
#[macro_export]
    macro_rules! test_ops_si {
        ($($tn:ident),+) => {
            $(
                test_arithmetic_!($tn, 0, 1, 2, 4);
                test_neg_!($tn, 0, 1, 2, 4);
                test_bit_arithmetic_!($tn);
            )+
        };
    }

#[cfg(test)]
#[macro_export]
    macro_rules! test_ops_ui {
        ($($tn:ident),+) => {
            $(
                test_arithmetic_!($tn, 0, 1, 2, 4);
                test_bit_arithmetic_!($tn);
            )+
        };
    }

#[cfg(test)]
#[macro_export]
    macro_rules! test_ops_f {
        ($($tn:ident),+)  => {
            $(
                test_arithmetic_!($tn, 0., 1., 2., 4.);
                test_neg_!($tn, 0., 1., 2., 4.);
            )+
        };
    }
