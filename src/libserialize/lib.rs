// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support code for encoding and decoding types.

/*
Core encoding and decoding interfaces.
*/

#![crate_name = "serialize"]
#![unstable(feature = "rustc_private",
            reason = "deprecated in favor of rustc-serialize on crates.io",
            issue = "27812")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       test(attr(allow(unused_variables), deny(warnings))))]
#![deny(warnings)]

#![feature(box_syntax)]
#![feature(collections)]
#![feature(core_intrinsics)]
#![feature(enumset)]
#![feature(specialization)]
#![feature(staged_api)]
#![cfg_attr(test, feature(test))]

extern crate collections;

extern crate rustc_i128;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable};

pub use self::serialize::{SpecializationError, SpecializedEncoder, SpecializedDecoder};
pub use self::serialize::{UseSpecializedEncodable, UseSpecializedDecodable};

mod serialize;
mod collection_impls;

pub mod hex;
pub mod json;

pub mod opaque;
pub mod leb128;

mod rustc_serialize {
    pub use serialize::*;
}
