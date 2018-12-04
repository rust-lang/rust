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
    feature(
        alloc,
        alloc_layout_extra,
        allocator_api,
        ptr_offset_from,
        test,
        core_intrinsics,
        dropck_eyepatch
    )
)]
#![warn(missing_docs)]

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
extern crate rand;

#[cfg(feature = "nightly")]
#[cfg_attr(test, macro_use)]
extern crate alloc;
extern crate byteorder;
#[cfg(feature = "rayon")]
extern crate rayon;
extern crate scopeguard;
#[cfg(feature = "serde")]
extern crate serde;
#[cfg(not(feature = "nightly"))]
#[cfg_attr(test, macro_use)]
extern crate std as alloc;

mod external_trait_impls;
mod fx;
mod map;
mod raw;
mod set;

pub mod hash_map {
    //! A hash map implemented with quadratic probing and SIMD lookup.
    pub use map::*;

    #[cfg(feature = "rayon")]
    /// [rayon]-based parallel iterator types for hash maps.
    /// You will rarely need to interact with it directly unless you have need
    /// to name one of the iterator types.
    ///
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    pub mod rayon {
        pub use external_trait_impls::rayon::map::*;
    }
}
pub mod hash_set {
    //! A hash set implemented as a `HashMap` where the value is `()`.
    pub use set::*;

    #[cfg(feature = "rayon")]
    /// [rayon]-based parallel iterator types for hash sets.
    /// You will rarely need to interact with it directly unless you have need
    /// to name one of the iterator types.
    ///
    /// [rayon]: https://docs.rs/rayon/1.0/rayon
    pub mod rayon {
        pub use external_trait_impls::rayon::set::*;
    }
}

pub use map::HashMap;
pub use set::HashSet;

/// Augments `AllocErr` with a CapacityOverflow variant.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CollectionAllocErr {
    /// Error due to the computed capacity exceeding the collection's maximum
    /// (usually `isize::MAX` bytes).
    CapacityOverflow,
    /// Error due to the allocator (see the `AllocErr` type's docs).
    AllocErr,
}
