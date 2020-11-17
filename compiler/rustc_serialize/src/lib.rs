//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(
    html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(box_syntax)]
#![feature(never_type)]
#![feature(nll)]
#![feature(associated_type_bounds)]
#![cfg_attr(bootstrap, feature(min_const_generics))]
#![cfg_attr(test, feature(test))]
#![allow(rustc::internal)]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

mod collection_impls;
mod serialize;

pub mod json;

pub mod leb128;
pub mod opaque;
