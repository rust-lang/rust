//! PowerPC Vector Scalar eXtensions (VSX) intrinsics.
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use crate::core_arch::powerpc::*;

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem::transmute;

types! {
    #![unstable(feature = "stdarch_powerpc", issue = "111145")]

    // pub struct vector_Float16 = f16x8;
    /// PowerPC-specific 128-bit wide vector of two packed `i64`
    pub struct vector_signed_long(2 x i64);
    /// PowerPC-specific 128-bit wide vector of two packed `u64`
    pub struct vector_unsigned_long(2 x u64);
    /// PowerPC-specific 128-bit wide vector mask of two `i64`
    pub struct vector_bool_long(2 x i64);
    /// PowerPC-specific 128-bit wide vector of two packed `f64`
    pub struct vector_double(2 x f64);
    // pub struct vector_signed_long_long = vector_signed_long;
    // pub struct vector_unsigned_long_long = vector_unsigned_long;
    // pub struct vector_bool_long_long = vector_bool_long;
    // pub struct vector_signed___int128 = i128x1;
    // pub struct vector_unsigned___int128 = i128x1;
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.ppc.altivec.vperm"]
    fn vperm(
        a: vector_signed_int,
        b: vector_signed_int,
        c: vector_unsigned_char,
    ) -> vector_signed_int;
}

mod sealed {
    use super::*;
    use crate::core_arch::simd::*;

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorPermDI {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        unsafe fn vec_xxpermdi(self, b: Self, dm: u8) -> Self;
    }

    // xxpermdi has an big-endian bias and extended mnemonics
    #[inline]
    #[target_feature(enable = "vsx")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(xxmrgld, dm = 0x0))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(xxspltd, dm = 0x0))]
    unsafe fn xxpermdi(a: vector_signed_long, b: vector_signed_long, dm: u8) -> vector_signed_long {
        let a: i64x2 = transmute(a);
        let b: i64x2 = transmute(b);
        let r: i64x2 = match dm & 0b11 {
            0 => simd_shuffle!(a, b, [0b00, 0b10]),
            1 => simd_shuffle!(a, b, [0b01, 0b10]),
            2 => simd_shuffle!(a, b, [0b00, 0b11]),
            _ => simd_shuffle!(a, b, [0b01, 0b11]),
        };
        transmute(r)
    }

    macro_rules! vec_xxpermdi {
        {$impl: ident} => {
            #[unstable(feature = "stdarch_powerpc", issue = "111145")]
            impl VectorPermDI for $impl {
                #[inline]
                #[target_feature(enable = "vsx")]
                unsafe fn vec_xxpermdi(self, b: Self, dm: u8) -> Self {
                    transmute(xxpermdi(transmute(self), transmute(b), dm))
                }
            }
        }
    }

    vec_xxpermdi! { vector_unsigned_long }
    vec_xxpermdi! { vector_signed_long }
    vec_xxpermdi! { vector_bool_long }
    vec_xxpermdi! { vector_double }

    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub trait VectorMergeEo {
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        unsafe fn vec_mergee(self, b: Self) -> Self;
        #[unstable(feature = "stdarch_powerpc", issue = "111145")]
        unsafe fn vec_mergeo(self, b: Self) -> Self;
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(
        all(test, target_endian = "little", target_feature = "power8-vector"),
        assert_instr(vmrgow)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", target_feature = "power8-vector"),
        assert_instr(vmrgew)
    )]
    unsafe fn mergee(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        let p = transmute(u8x16::new(
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B, 0x18, 0x19,
            0x1A, 0x1B,
        ));
        vec_perm(a, b, p)
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(
        all(test, target_endian = "little", target_feature = "power8-vector"),
        assert_instr(vmrgew)
    )]
    #[cfg_attr(
        all(test, target_endian = "big", target_feature = "power8-vector"),
        assert_instr(vmrgow)
    )]
    unsafe fn mergeo(a: vector_signed_int, b: vector_signed_int) -> vector_signed_int {
        let p = transmute(u8x16::new(
            0x04, 0x05, 0x06, 0x07, 0x14, 0x15, 0x16, 0x17, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D,
            0x1E, 0x1F,
        ));
        vec_perm(a, b, p)
    }

    macro_rules! vec_mergeeo {
        { $impl: ident, $even: ident, $odd: ident } => {
            #[unstable(feature = "stdarch_powerpc", issue = "111145")]
            impl VectorMergeEo for $impl {
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_mergee(self, b: Self) -> Self {
                    transmute(mergee(transmute(self), transmute(b)))
                }
                #[inline]
                #[target_feature(enable = "altivec")]
                unsafe fn vec_mergeo(self, b: Self) -> Self {
                    transmute(mergeo(transmute(self), transmute(b)))
                }
            }
        }
    }

    vec_mergeeo! { vector_signed_int, mergee, mergeo }
    vec_mergeeo! { vector_unsigned_int, mergee, mergeo }
    vec_mergeeo! { vector_bool_int, mergee, mergeo }
    vec_mergeeo! { vector_float, mergee, mergeo }
}

/// Vector permute.
#[inline]
#[target_feature(enable = "vsx")]
//#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_xxpermdi<T, const DM: i32>(a: T, b: T) -> T
where
    T: sealed::VectorPermDI,
{
    static_assert_uimm_bits!(DM, 2);
    a.vec_xxpermdi(b, DM as u8)
}

/// Vector Merge Even
///
/// ## Purpose
/// Merges the even-numbered values from two vectors.
///
/// ## Result value
/// The even-numbered elements of a are stored into the even-numbered elements of r.
/// The even-numbered elements of b are stored into the odd-numbered elements of r.
#[inline]
#[target_feature(enable = "altivec")]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_mergee<T>(a: T, b: T) -> T
where
    T: sealed::VectorMergeEo,
{
    a.vec_mergee(b)
}

/// Vector Merge Odd
///
/// ## Purpose
/// Merges the odd-numbered values from two vectors.
///
/// ## Result value
/// The odd-numbered elements of a are stored into the even-numbered elements of r.
/// The odd-numbered elements of b are stored into the odd-numbered elements of r.
#[inline]
#[target_feature(enable = "altivec")]
#[unstable(feature = "stdarch_powerpc", issue = "111145")]
pub unsafe fn vec_mergeo<T>(a: T, b: T) -> T
where
    T: sealed::VectorMergeEo,
{
    a.vec_mergeo(b)
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "powerpc")]
    use crate::core_arch::arch::powerpc::*;

    #[cfg(target_arch = "powerpc64")]
    use crate::core_arch::arch::powerpc64::*;

    use crate::core_arch::simd::*;
    use crate::mem::transmute;
    use stdarch_test::simd_test;

    macro_rules! test_vec_xxpermdi {
        {$name:ident, $shorttype:ident, $longtype:ident, [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "vsx")]
            unsafe fn $name() {
                let a: $longtype = transmute($shorttype::new($($a),+, $($b),+));
                let b = transmute($shorttype::new($($c),+, $($d),+));

                assert_eq!($shorttype::new($($a),+, $($c),+), transmute(vec_xxpermdi::<_, 0>(a, b)));
                assert_eq!($shorttype::new($($b),+, $($c),+), transmute(vec_xxpermdi::<_, 1>(a, b)));
                assert_eq!($shorttype::new($($a),+, $($d),+), transmute(vec_xxpermdi::<_, 2>(a, b)));
                assert_eq!($shorttype::new($($b),+, $($d),+), transmute(vec_xxpermdi::<_, 3>(a, b)));
            }
        }
    }

    test_vec_xxpermdi! {test_vec_xxpermdi_u64x2, u64x2, vector_unsigned_long, [0], [1], [2], [3]}
    test_vec_xxpermdi! {test_vec_xxpermdi_i64x2, i64x2, vector_signed_long, [0], [-1], [2], [-3]}
    test_vec_xxpermdi! {test_vec_xxpermdi_m64x2, m64x2, vector_bool_long, [false], [true], [false], [true]}
    test_vec_xxpermdi! {test_vec_xxpermdi_f64x2, f64x2, vector_double, [0.0], [1.0], [2.0], [3.0]}
}
