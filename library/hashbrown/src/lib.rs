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
    feature(alloc, allocator_api, ptr_offset_from, test, core_intrinsics)
)]
#![warn(missing_docs)]

#[cfg(feature = "nightly")]
extern crate alloc;
extern crate byteorder;
extern crate scopeguard;
#[cfg(not(feature = "nightly"))]
extern crate std as alloc;

mod fx;
mod map;
mod raw;
mod set;

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
