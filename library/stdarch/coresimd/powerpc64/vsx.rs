//! PowerPC Vectir Scalar eXtensions (VSX) intrinsics.
//!
//! The references are: [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA
//! NVlink)] and [POWER ISA v3.0B (for POWER9)].
//!
//! [POWER ISA v2.07B (for POWER8 & POWER8 with NVIDIA NVlink)]: https://ibm.box.com/s/jd5w15gz301s5b5dt375mshpq9c3lh4u
//! [POWER ISA v3.0B (for POWER9)]: https://ibm.box.com/s/1hzcwkwf8rbju5h9iyf44wm94amnlcrv

#![allow(non_camel_case_types)]

use coresimd::simd::*;

// pub type vector_Float16 = f16x8;
pub type vector_signed_long = i64x2;
pub type vector_unsigned_long = u64x2;
pub type vector_bool_long = u64x2;
pub type vector_signed_long_long = vector_signed_long;
pub type vector_unsigned_long_long = vector_unsigned_long;
pub type vector_bool_long_long = vector_bool_long;
pub type vector_double = f64x2;
// pub type vector_signed___int128 = i128x1;
// pub type vector_unsigned___int128 = i128x1;
