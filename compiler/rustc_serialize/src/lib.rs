//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(never_type)]
#![feature(associated_type_bounds)]
#![feature(min_specialization)]
#![feature(core_intrinsics)]
#![feature(maybe_uninit_slice)]
#![feature(let_else)]
#![feature(new_uninit)]
#![feature(allocator_api)]
#![cfg_attr(test, feature(test))]
#![allow(rustc::internal)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod collection_impls;
mod serialize;

pub mod leb128;
pub mod opaque;
