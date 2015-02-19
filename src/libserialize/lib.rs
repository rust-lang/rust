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
            reason = "deprecated in favor of rustc-serialize on crates.io")]
#![staged_api]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]

#![feature(box_syntax)]
#![feature(collections)]
#![feature(core)]
#![feature(int_uint)]
#![feature(old_io)]
#![feature(old_path)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(std_misc)]
#![feature(unicode)]
#![cfg_attr(test, feature(test))]

// test harness access
#[cfg(test)] extern crate test;
#[macro_use] extern crate log;

extern crate unicode;
extern crate collections;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable,
                          DecoderHelpers, EncoderHelpers};

mod serialize;
mod collection_impls;

pub mod hex;
pub mod json;

mod rustc_serialize {
    pub use serialize::*;
}
