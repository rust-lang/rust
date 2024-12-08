#[macro_use]
mod swizzle;

mod alias;
mod cast;
mod fmt;
mod iter;
mod lane_count;
mod masks;
mod ops;
mod select;
mod swizzle_dyn;
mod to_bytes;
mod vector;
mod vendor;

pub mod simd {
    #![doc = include_str!("core_simd_docs.md")]

    pub mod prelude;

    pub mod num;

    pub mod ptr;

    pub mod cmp;

    pub use crate::core_simd::alias::*;
    pub use crate::core_simd::cast::*;
    pub use crate::core_simd::lane_count::{LaneCount, SupportedLaneCount};
    pub use crate::core_simd::masks::*;
    pub use crate::core_simd::swizzle::*;
    pub use crate::core_simd::to_bytes::ToBytes;
    pub use crate::core_simd::vector::*;
}
