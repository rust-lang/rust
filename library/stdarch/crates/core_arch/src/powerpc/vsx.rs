//! PowerPC Vector Scalar eXtensions (VSX) intrinsics.
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use crate::core_arch::simd_llvm::*;

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::mem;

types! {
    // pub struct vector_Float16 = f16x8;
    /// PowerPC-specific 128-bit wide vector of two packed `i64`
    pub struct vector_signed_long(i64, i64);
    /// PowerPC-specific 128-bit wide vector of two packed `u64`
    pub struct vector_unsigned_long(u64, u64);
    /// PowerPC-specific 128-bit wide vector mask of two elements
    pub struct vector_bool_long(i64, i64);
    /// PowerPC-specific 128-bit wide vector of two packed `f64`
    pub struct vector_double(f64, f64);
    // pub struct vector_signed_long_long = vector_signed_long;
    // pub struct vector_unsigned_long_long = vector_unsigned_long;
    // pub struct vector_bool_long_long = vector_bool_long;
    // pub struct vector_signed___int128 = i128x1;
    // pub struct vector_unsigned___int128 = i128x1;
}

mod sealed {
    use super::*;
    use crate::core_arch::simd::*;

    pub trait VectorPermDI {
        unsafe fn vec_xxpermdi(self, b: Self, dm: u8) -> Self;
    }

    // xxpermdi has an big-endian bias and extended mnemonics
    #[inline]
    #[target_feature(enable = "vsx")]
    #[cfg_attr(all(test, target_endian = "little"), assert_instr(xxmrgld, dm = 0x0))]
    #[cfg_attr(all(test, target_endian = "big"), assert_instr(xxspltd, dm = 0x0))]
    unsafe fn xxpermdi(a: i64x2, b: i64x2, dm: u8) -> i64x2 {
        match dm & 0b11 {
            0 => simd_shuffle2!(a, b, [0b00, 0b10]),
            1 => simd_shuffle2!(a, b, [0b01, 0b10]),
            2 => simd_shuffle2!(a, b, [0b00, 0b11]),
            _ => simd_shuffle2!(a, b, [0b01, 0b11]),
        }
    }

    macro_rules! vec_xxpermdi {
        {$impl: ident} => {
            impl VectorPermDI for $impl {
                #[inline]
                #[target_feature(enable = "vsx")]
                unsafe fn vec_xxpermdi(self, b: Self, dm: u8) -> Self {
                    mem::transmute(xxpermdi(mem::transmute(self), mem::transmute(b), dm))
                }
            }
        }
    }

    vec_xxpermdi! { vector_unsigned_long }
    vec_xxpermdi! { vector_signed_long }
    vec_xxpermdi! { vector_bool_long }
    vec_xxpermdi! { vector_double }
}

/// Vector permute.
#[inline]
#[target_feature(enable = "vsx")]
//#[rustc_legacy_const_generics(2)]
pub unsafe fn vec_xxpermdi<T, const DM: i32>(a: T, b: T) -> T
where
    T: sealed::VectorPermDI,
{
    static_assert_imm2!(DM);
    a.vec_xxpermdi(b, DM as u8)
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "powerpc")]
    use crate::core_arch::arch::powerpc::*;

    #[cfg(target_arch = "powerpc64")]
    use crate::core_arch::arch::powerpc64::*;

    use super::mem;
    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    macro_rules! test_vec_xxpermdi {
        {$name:ident, $shorttype:ident, $longtype:ident, [$($a:expr),+], [$($b:expr),+], [$($c:expr),+], [$($d:expr),+]} => {
            #[simd_test(enable = "vsx")]
            unsafe fn $name() {
                let a: $longtype = mem::transmute($shorttype::new($($a),+, $($b),+));
                let b = mem::transmute($shorttype::new($($c),+, $($d),+));

                assert_eq!($shorttype::new($($a),+, $($c),+), mem::transmute(vec_xxpermdi::<_, 0>(a, b)));
                assert_eq!($shorttype::new($($b),+, $($c),+), mem::transmute(vec_xxpermdi::<_, 1>(a, b)));
                assert_eq!($shorttype::new($($a),+, $($d),+), mem::transmute(vec_xxpermdi::<_, 2>(a, b)));
                assert_eq!($shorttype::new($($b),+, $($d),+), mem::transmute(vec_xxpermdi::<_, 3>(a, b)));
            }
        }
    }

    test_vec_xxpermdi! {test_vec_xxpermdi_u64x2, u64x2, vector_unsigned_long, [0], [1], [2], [3]}
    test_vec_xxpermdi! {test_vec_xxpermdi_i64x2, i64x2, vector_signed_long, [0], [-1], [2], [-3]}
    test_vec_xxpermdi! {test_vec_xxpermdi_m64x2, m64x2, vector_bool_long, [false], [true], [false], [true]}
    test_vec_xxpermdi! {test_vec_xxpermdi_f64x2, f64x2, vector_double, [0.0], [1.0], [2.0], [3.0]}
}
