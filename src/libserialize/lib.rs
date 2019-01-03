//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(
    html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
    html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
    html_root_url = "https://doc.rust-lang.org/nightly/",
    html_playground_url = "https://play.rust-lang.org/",
    test(attr(allow(unused_variables), deny(warnings)))
)]
#![feature(box_syntax)]
#![feature(core_intrinsics)]
#![feature(specialization)]
#![feature(never_type)]
#![feature(nll)]
#![cfg_attr(test, feature(test))]

pub use self::serialize::{Decodable, Decoder, Encodable, Encoder};

pub use self::serialize::{SpecializationError, SpecializedDecoder, SpecializedEncoder};
pub use self::serialize::{UseSpecializedDecodable, UseSpecializedEncodable};

extern crate smallvec;

mod collection_impls;
mod serialize;

pub mod hex;
pub mod json;

pub mod leb128;
pub mod opaque;

mod rustc_serialize {
    pub use serialize::*;
}
