//! Support code for encoding and decoding types.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![cfg_attr(not(bootstrap), doc(rust_logo))]
#![cfg_attr(not(bootstrap), allow(internal_features))]
#![cfg_attr(not(bootstrap), feature(rustdoc_internals))]
#![feature(allocator_api)]
#![feature(associated_type_bounds)]
#![feature(const_option)]
#![feature(core_intrinsics)]
#![feature(inline_const)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(ptr_sub_ptr)]
#![feature(slice_first_last_chunk)]
#![cfg_attr(test, feature(test))]
#![allow(rustc::internal)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod serialize;

pub mod leb128;
pub mod opaque;
