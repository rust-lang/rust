//! Support code for encoding and decoding types.

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![doc(rust_logo)]
#![allow(internal_features)]
#![feature(rustdoc_internals)]
#![feature(associated_type_bounds)]
#![feature(const_option)]
#![feature(core_intrinsics)]
#![feature(inline_const)]
#![feature(min_specialization)]
#![feature(never_type)]
#![feature(ptr_sub_ptr)]
#![cfg_attr(test, feature(test))]
#![allow(rustc::internal)]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod serialize;

pub mod leb128;
pub mod opaque;
