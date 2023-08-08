#[macro_use]
mod swizzle;

pub(crate) mod intrinsics;

#[cfg(feature = "generic_const_exprs")]
mod to_bytes;

mod alias;
mod cast;
mod elements;
mod eq;
mod fmt;
mod iter;
mod lane_count;
mod masks;
mod ops;
mod ord;
mod select;
mod swizzle_dyn;
mod vector;
mod vendor;

#[doc = include_str!("core_simd_docs.md")]
pub mod simd {
    pub mod prelude;

    pub(crate) use crate::core_simd::intrinsics;

    pub use crate::core_simd::alias::*;
    pub use crate::core_simd::cast::*;
    pub use crate::core_simd::elements::*;
    pub use crate::core_simd::eq::*;
    pub use crate::core_simd::lane_count::{LaneCount, SupportedLaneCount};
    pub use crate::core_simd::masks::*;
    pub use crate::core_simd::ord::*;
    pub use crate::core_simd::swizzle::*;
    pub use crate::core_simd::swizzle_dyn::*;
    pub use crate::core_simd::vector::*;
}
