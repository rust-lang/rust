//! Support code for encoding and decoding types.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::internal)]
#![cfg_attr(not(bootstrap), feature(sized_hierarchy))]
#![cfg_attr(test, feature(test))]
#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![doc(rust_logo)]
#![feature(core_intrinsics)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

// Allows macros to refer to this crate as `::rustc_serialize`.
#[cfg(test)]
extern crate self as rustc_serialize;

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod serialize;

pub mod int_overflow;
pub mod leb128;
pub mod opaque;

// This has nothing to do with `rustc_serialize` but it is convenient to define it in one place
// for the rest of the compiler so that `cfg(bootstrap)` doesn't need to be littered throughout
// the compiler wherever `PointeeSized` would be used. `rustc_serialize` happens to be the deepest
// crate in the crate graph which uses `PointeeSized`.
//
// When bootstrap bumps, remove both the `cfg(not(bootstrap))` and `cfg(bootstrap)` lines below
// and just import `std::marker::PointeeSized` whereever this item was used.

#[cfg(not(bootstrap))]
pub use std::marker::PointeeSized;

#[cfg(bootstrap)]
pub trait PointeeSized {}
#[cfg(bootstrap)]
impl<T: ?Sized> PointeeSized for T {}
