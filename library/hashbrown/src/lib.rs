//! This crate is a Rust port of Google's high-performance [SwissTable] hash
//! map, adapted to make it a drop-in replacement for Rust's standard `HashMap`
//! and `HashSet` types.
//!
//! The original C++ version of [SwissTable] can be found [here], and this
//! [CppCon talk] gives an overview of how the algorithm works.
//!
//! [SwissTable]: https://abseil.io/blog/20180927-swisstables
//! [here]: https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h
//! [CppCon talk]: https://www.youtube.com/watch?v=ncHmEUmJZf4

#![no_std]
#![cfg_attr(
    feature = "nightly",
    feature(
        alloc_layout_extra,
        allocator_api,
        ptr_offset_from,
        test,
        core_intrinsics,
        dropck_eyepatch
    )
)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![warn(rust_2018_idioms)]

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(feature = "nightly")]
#[cfg_attr(test, macro_use)]
extern crate alloc;
#[cfg(not(feature = "nightly"))]
extern crate std as alloc;

#[macro_use]
mod macros;

mod external_trait_impls;
mod fx;
mod map;
mod raw;
#[cfg(feature = "rustc-dep-of-std")]
mod rustc_entry;
mod scopeguard;
mod set;

pub mod hash_map {
    //! A hash map implemented with quadratic probing and SIMD lookup.
    pub use crate::map::*;

    #[cfg(feature = "rustc-dep-of-std")]
    pub use crate::rustc_entry::*;

    #[cfg(feature = "rayon")]
    /// [rayon]-based parallel iterator types for hash maps.
    /// You will rarely need to interact with it directly unless you have need
    /// to name one of the iterator types.
    ///
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    pub mod rayon {
        pub use crate::external_trait_impls::rayon::map::*;
    }
}
pub mod hash_set {
    //! A hash set implemented as a `HashMap` where the value is `()`.
    pub use crate::set::*;

    #[cfg(feature = "rayon")]
    /// [rayon]-based parallel iterator types for hash sets.
    /// You will rarely need to interact with it directly unless you have need
    /// to name one of the iterator types.
    ///
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    pub mod rayon {
        pub use crate::external_trait_impls::rayon::set::*;
    }
}

pub use crate::map::HashMap;
pub use crate::set::HashSet;

/// Augments `AllocErr` with a `CapacityOverflow` variant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CollectionAllocErr {
    /// Error due to the computed capacity exceeding the collection's maximum
    /// (usually `isize::MAX` bytes).
    CapacityOverflow,
    /// Error due to the allocator (see the `AllocErr` type's docs).
    AllocErr,
}
