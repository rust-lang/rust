//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
// tidy-alphabetical-start
#![allow(rustc::internal)]
#![deny(rustc::diagnostic_outside_of_impl)]
#![deny(rustc::untranslatable_diagnostic)]
#![feature(allocator_api)]
#![feature(associated_type_bounds)]
#![feature(core_intrinsics)]
#![feature(maybe_uninit_slice)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(new_uninit)]
#![feature(ptr_sub_ptr)]
// tidy-alphabetical-end
#![cfg_attr(test, feature(test))]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod collection_impls;
mod serialize;

pub mod leb128;
pub mod opaque;
