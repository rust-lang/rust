//! PowerPC Vectir Scalar eXtensions (VSX) intrinsics.
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use coresimd::powerpc::*;
use coresimd::simd::*;

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

impl_from_bits_!(
    vector_signed_long: u64x2,
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
    vector_bool_int,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);
impl_from_bits_!(
    i64x2:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int,
    vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_unsigned_long: u64x2,
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
    vector_bool_int,
    vector_signed_long,
    vector_bool_long,
    vector_double
);
impl_from_bits_!(
    u64x2:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int,
    vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_double: u64x2,
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
    vector_bool_int,
    vector_signed_long,
    vector_unsigned_long,
    vector_bool_long
);
impl_from_bits_!(
    f64x2:
    vector_signed_char,
    vector_unsigned_char,
    vector_bool_char,
    vector_signed_short,
    vector_unsigned_short,
    vector_bool_short,
    vector_signed_int,
    vector_unsigned_int,
    vector_float,
    vector_bool_int,
    vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(vector_bool_long: m64x2);
impl_from_bits_!(m64x2: vector_bool_long);
impl_from_bits_!(m32x4: vector_bool_long);
impl_from_bits_!(m16x8: vector_bool_long);
impl_from_bits_!(m8x16: vector_bool_long);
impl_from_bits_!(vector_bool_char: vector_bool_long);
impl_from_bits_!(vector_bool_short: vector_bool_long);
impl_from_bits_!(vector_bool_int: vector_bool_long);

impl_from_bits_!(
    vector_signed_char: vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_unsigned_char: vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_signed_short: vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_unsigned_short: vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_signed_int: vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);

impl_from_bits_!(
    vector_unsigned_int: vector_signed_long,
    vector_unsigned_long,
    vector_bool_long,
    vector_double
);
