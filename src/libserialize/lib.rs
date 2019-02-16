//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       test(attr(allow(unused_variables), deny(warnings))))]

#![deny(rust_2018_idioms)]

#![feature(box_syntax)]
#![feature(core_intrinsics)]
#![feature(specialization)]
#![feature(never_type)]
#![feature(nll)]
#![cfg_attr(test, feature(test))]

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable};

pub use self::serialize::{SpecializationError, SpecializedEncoder, SpecializedDecoder};
pub use self::serialize::{UseSpecializedEncodable, UseSpecializedDecodable};

mod serialize;
mod collection_impls;

pub mod hex;
pub mod json;

pub mod opaque;
pub mod leb128;
