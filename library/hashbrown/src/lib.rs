//! This crate is a Rust port of Google's high-performance [SwissTable] hash
//! map, adapted to make it a drop-in replacement for Rust's standard `HashMap`
//! and `HashSet` types.
//!
//! The original C++ version of SwissTable can be found [here], and this
//! [CppCon talk] gives an overview of how the algorithm works.
//!
//! [SwissTable]: https://abseil.io/blog/20180927-swisstables
//! [here]: https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h
//! [CppCon talk]: https://www.youtube.com/watch?v=ncHmEUmJZf4

#![no_std]
#![cfg_attr(
    feature = "nightly",
    feature(alloc, alloc_layout_extra, allocator_api, ptr_offset_from, test, core_intrinsics)
)]
#![warn(missing_docs)]

#[cfg(feature = "nightly")]
extern crate alloc;
extern crate byteorder;
extern crate scopeguard;
#[cfg(feature = "serde")]
extern crate serde;
#[cfg(not(feature = "nightly"))]
extern crate std as alloc;

mod fx;
mod map;
mod raw;
mod set;

#[cfg(feature = "serde")]
mod size_hint {
    use core::cmp;

    /// This presumably exists to prevent denial of service attacks.
    ///
    /// Original discussion: https://github.com/serde-rs/serde/issues/1114.
    #[inline]
    pub(crate) fn cautious(hint: Option<usize>) -> usize {
        cmp::min(hint.unwrap_or(0), 4096)
    }
}

pub mod hash_map {
    //! A hash map implemented with quadratic probing and SIMD lookup.
    pub use map::*;
}
pub mod hash_set {
    //! A hash set implemented as a `HashMap` where the value is `()`.
    pub use set::*;
}
pub use map::HashMap;
pub use set::HashSet;
