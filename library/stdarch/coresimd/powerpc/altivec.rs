//! PowerPC AltiVec intrinsics.
//!
//! AltiVec is a brandname trademarked by Freescale (previously Motorola) for
//! the standard `Category:Vector` part of the Power ISA v.2.03 specification.
//! This Category is also known as VMX (used by IBM), and "Velocity Engine" (a
//! brand name previously used by Apple).
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use coresimd::simd::*;
use coresimd::simd_llvm::*;
#[cfg(test)]
use stdsimd_test::assert_instr;

types! {
    /// PowerPC-specific 128-bit wide vector of sixteen packed `i8`
    pub struct vector_signed_char(i8, i8, i8, i8, i8, i8, i8, i8,
                                  i8, i8, i8, i8, i8, i8, i8, i8);
    /// PowerPC-specific 128-bit wide vector of sixteen packed `u8`
    pub struct vector_unsigned_char(u8, u8, u8, u8, u8, u8, u8, u8,
                                    u8, u8, u8, u8, u8, u8, u8, u8);

    /// PowerPC-specific 128-bit wide vector mask of sixteen packed elements
    pub struct vector_bool_char(i8, i8, i8, i8, i8, i8, i8, i8,
                                i8, i8, i8, i8, i8, i8, i8, i8);
    /// PowerPC-specific 128-bit wide vector of eight packed `i16`
    pub struct vector_signed_short(i16, i16, i16, i16, i16, i16, i16, i16);
    /// PowerPC-specific 128-bit wide vector of eight packed `u16`
    pub struct vector_unsigned_short(u16, u16, u16, u16, u16, u16, u16, u16);
    /// PowerPC-specific 128-bit wide vector mask of eight packed elements
    pub struct vector_bool_short(i16, i16, i16, i16, i16, i16, i16, i16);
    // pub struct vector_pixel(???);
    /// PowerPC-specific 128-bit wide vector of four packed `i32`
    pub struct vector_signed_int(i32, i32, i32, i32);
    /// PowerPC-specific 128-bit wide vector of four packed `u32`
    pub struct vector_unsigned_int(u32, u32, u32, u32);
    /// PowerPC-specific 128-bit wide vector mask of four packed elements
    pub struct vector_bool_int(i32, i32, i32, i32);
    /// PowerPC-specific 128-bit wide vector of four packed `f32`
    pub struct vector_float(f32, f32, f32, f32);
}

impl_from_bits_!(
    vector_signed_char: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);
impl_from_bits_!(
    i8x16:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

impl_from_bits_!(
    vector_unsigned_char: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_signed_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);
impl_from_bits_!(
    u8x16:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

impl_from_bits_!(
    vector_bool_char: m64x2,
    m32x4,
    m16x8,
    m8x16,
    vector_bool_short,
    vector_bool_int
);
impl_from_bits_!(
    m8x16: vector_bool_char,
    vector_bool_short,
    vector_bool_int
);

impl_from_bits_!(
    vector_signed_short: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_signed_char,
    vector_bool_char,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);
impl_from_bits_!(
    i16x8:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

impl_from_bits_!(
    vector_unsigned_short: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_signed_char,
    vector_bool_char,
    vector_signed_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);
impl_from_bits_!(
    u16x8:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

impl_from_bits_!(
    vector_bool_short: m64x2,
    m32x4,
    m16x8,
    m8x16,
    vector_bool_int
);
impl_from_bits_!(m16x8: vector_bool_short, vector_bool_int);

impl_from_bits_!(
    vector_signed_int: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_signed_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);
impl_from_bits_!(
    i32x4:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

impl_from_bits_!(
    vector_unsigned_int: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_signed_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_float,
    vector_bool_int
);
impl_from_bits_!(
    u32x4:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

impl_from_bits_!(
    vector_bool_int: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16
);
impl_from_bits_!(m32x4: vector_bool_int);

impl_from_bits_!(
    vector_float: u64x2,
    i64x2,
    f64x2,
    m64x2,
    u32x4,
    i32x4,
    f32x4,
    m32x4,
    u16x8,
    i16x8,
    m16x8,
    u8x16,
    i8x16,
    m8x16,
    vector_signed_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_bool_int
);
impl_from_bits_!(
    f32x4:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int
);

mod sealed {

    use super::*;

    pub trait VectorAdd<Other> {
        type Result;
        unsafe fn vec_add(self, other: Other) -> Self::Result;
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_bc_sc(
        a: vector_bool_char, b: vector_signed_char,
    ) -> vector_signed_char {
        simd_add(a.into_bits(), b)
    }
    impl VectorAdd<vector_signed_char> for vector_bool_char {
        type Result = vector_signed_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_char) -> Self::Result {
            vec_add_bc_sc(self, other)
        }
    }
    impl VectorAdd<vector_bool_char> for vector_signed_char {
        type Result = vector_signed_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_char) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_sc_sc(
        a: vector_signed_char, b: vector_signed_char,
    ) -> vector_signed_char {
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_char> for vector_signed_char {
        type Result = vector_signed_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_char) -> Self::Result {
            vec_add_sc_sc(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_bc_uc(
        a: vector_bool_char, b: vector_unsigned_char,
    ) -> vector_unsigned_char {
        simd_add(a.into_bits(), b)
    }
    impl VectorAdd<vector_unsigned_char> for vector_bool_char {
        type Result = vector_unsigned_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_char) -> Self::Result {
            vec_add_bc_uc(self, other)
        }
    }
    impl VectorAdd<vector_bool_char> for vector_unsigned_char {
        type Result = vector_unsigned_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_char) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vaddubm))]
    pub unsafe fn vec_add_uc_uc(
        a: vector_unsigned_char, b: vector_unsigned_char,
    ) -> vector_unsigned_char {
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_char> for vector_unsigned_char {
        type Result = vector_unsigned_char;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_char) -> Self::Result {
            vec_add_uc_uc(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_bs_ss(
        a: vector_bool_short, b: vector_signed_short,
    ) -> vector_signed_short {
        let a: i16x8 = a.into_bits();
        let a: vector_signed_short = simd_cast(a);
        simd_add(a, b)
    }

    impl VectorAdd<vector_signed_short> for vector_bool_short {
        type Result = vector_signed_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_short) -> Self::Result {
            vec_add_bs_ss(self, other)
        }
    }
    impl VectorAdd<vector_bool_short> for vector_signed_short {
        type Result = vector_signed_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_short) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_ss_ss(
        a: vector_signed_short, b: vector_signed_short,
    ) -> vector_signed_short {
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_short> for vector_signed_short {
        type Result = vector_signed_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_short) -> Self::Result {
            vec_add_ss_ss(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_bs_us(
        a: vector_bool_short, b: vector_unsigned_short,
    ) -> vector_unsigned_short {
        let a: i16x8 = a.into_bits();
        let a: vector_unsigned_short = simd_cast(a);
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_short> for vector_bool_short {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_short) -> Self::Result {
            vec_add_bs_us(self, other)
        }
    }
    impl VectorAdd<vector_bool_short> for vector_unsigned_short {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_short) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduhm))]
    pub unsafe fn vec_add_us_us(
        a: vector_unsigned_short, b: vector_unsigned_short,
    ) -> vector_unsigned_short {
        simd_add(a, b)
    }

    impl VectorAdd<vector_unsigned_short> for vector_unsigned_short {
        type Result = vector_unsigned_short;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_short) -> Self::Result {
            vec_add_us_us(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_bi_si(
        a: vector_bool_int, b: vector_signed_int,
    ) -> vector_signed_int {
        let a: i32x4 = a.into_bits();
        let a: vector_signed_int = simd_cast(a);
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_int> for vector_bool_int {
        type Result = vector_signed_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_int) -> Self::Result {
            vec_add_bi_si(self, other)
        }
    }
    impl VectorAdd<vector_bool_int> for vector_signed_int {
        type Result = vector_signed_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_int) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_si_si(
        a: vector_signed_int, b: vector_signed_int,
    ) -> vector_signed_int {
        simd_add(a, b)
    }
    impl VectorAdd<vector_signed_int> for vector_signed_int {
        type Result = vector_signed_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_signed_int) -> Self::Result {
            vec_add_si_si(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_bi_ui(
        a: vector_bool_int, b: vector_unsigned_int,
    ) -> vector_unsigned_int {
        let a: i32x4 = a.into_bits();
        let a: vector_unsigned_int = simd_cast(a);
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_int> for vector_bool_int {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_int) -> Self::Result {
            vec_add_bi_ui(self, other)
        }
    }
    impl VectorAdd<vector_bool_int> for vector_unsigned_int {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_bool_int) -> Self::Result {
            other.vec_add(self)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(vadduwm))]
    pub unsafe fn vec_add_ui_ui(
        a: vector_unsigned_int, b: vector_unsigned_int,
    ) -> vector_unsigned_int {
        simd_add(a, b)
    }
    impl VectorAdd<vector_unsigned_int> for vector_unsigned_int {
        type Result = vector_unsigned_int;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_unsigned_int) -> Self::Result {
            vec_add_ui_ui(self, other)
        }
    }

    #[inline]
    #[target_feature(enable = "altivec")]
    #[cfg_attr(test, assert_instr(xvaddsp))]
    pub unsafe fn vec_add_float_float(
        a: vector_float, b: vector_float,
    ) -> vector_float {
        simd_add(a, b)
    }

    impl VectorAdd<vector_float> for vector_float {
        type Result = vector_float;
        #[inline]
        #[target_feature(enable = "altivec")]
        unsafe fn vec_add(self, other: vector_float) -> Self::Result {
            vec_add_float_float(self, other)
        }
    }
}

/// Vector add.
#[inline]
#[target_feature(enable = "altivec")]
pub unsafe fn vec_add<T, U>(a: T, b: U) -> <T as sealed::VectorAdd<U>>::Result
where
    T: sealed::VectorAdd<U>,
{
    a.vec_add(b)
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "powerpc")]
    use coresimd::arch::powerpc::*;

    #[cfg(target_arch = "powerpc64")]
    use coresimd::arch::powerpc64::*;

    use simd::*;
    use stdsimd_test::simd_test;

    #[simd_test(enable = "altivec")]
    unsafe fn vec_add_i32x4_i32x4() {
        let x = i32x4::new(1, 2, 3, 4);
        let y = i32x4::new(4, 3, 2, 1);
        let x: vector_signed_int = x.into_bits();
        let y: vector_signed_int = y.into_bits();
        let z = vec_add(x, y);
        assert_eq!(i32x4::splat(5), z.into_bits());
    }
}
