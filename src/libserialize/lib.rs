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
#![experimental]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/master/",
       html_playground_url = "http://play.rust-lang.org/")]
#![allow(unknown_features)]
#![feature(macro_rules, managed_boxes, default_type_params, phase, slicing_syntax)]

// test harness access
#[cfg(test)]
extern crate test;

#[phase(plugin, link)]
extern crate log;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable,
                          DecoderHelpers, EncoderHelpers};

mod serialize;
mod collection_impls;

pub mod base64;
pub mod hex;
pub mod json;
