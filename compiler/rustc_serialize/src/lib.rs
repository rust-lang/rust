//! Support code for encoding and decoding types.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::internal)]
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
#![feature(ptr_sub_ptr)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod serialize;

pub mod int_overflow;
pub mod leb128;
pub mod opaque;
