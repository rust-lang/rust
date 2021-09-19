#[macro_use]
mod permute;
#[macro_use]
mod reduction;

mod select;

#[cfg(feature = "generic_const_exprs")]
mod to_bytes;

mod comparisons;
mod fmt;
mod intrinsics;
mod iter;
mod math;
mod ops;
mod round;
mod vendor;

mod lane_count;

mod masks;

mod vector;

#[doc = include_str!("core_simd_docs.md")]
pub mod simd {
    pub use crate::core_simd::lane_count::*;
    pub use crate::core_simd::masks::*;
    pub use crate::core_simd::select::Select;
    pub use crate::core_simd::vector::*;
    pub(crate) use crate::core_simd::*;
}
