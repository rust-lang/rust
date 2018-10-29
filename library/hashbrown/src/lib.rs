//! A high-performance replacement for the standard library `HashMap`.
//!
//! The API of this crate mirrors that of the hash table implementation in
//! `std::collections`.

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
    //! A hash map implemented with linear probing and Robin Hood bucket stealing.
    pub use map::*;
}
pub mod hash_set {
    //! A hash set implemented as a `HashMap` where the value is `()`.
    pub use set::*;
}
pub use map::HashMap;
pub use set::HashSet;
