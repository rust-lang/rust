#[macro_use]
mod reduction;

#[macro_use]
mod swizzle;

pub(crate) mod intrinsics;

#[cfg(feature = "generic_const_exprs")]
mod to_bytes;

mod comparisons;
mod fmt;
mod iter;
mod lane_count;
mod masks;
mod math;
mod ops;
mod round;
mod select;
mod vector;
mod vendor;

#[doc = include_str!("core_simd_docs.md")]
pub mod simd {
    pub(crate) use crate::core_simd::intrinsics;

    pub use crate::core_simd::lane_count::{LaneCount, SupportedLaneCount};
    pub use crate::core_simd::masks::*;
    pub use crate::core_simd::select::Select;
    pub use crate::core_simd::swizzle::*;
    pub use crate::core_simd::vector::*;
}
