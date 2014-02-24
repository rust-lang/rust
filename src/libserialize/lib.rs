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

#[crate_id = "serialize#0.10-pre"];
#[crate_type = "rlib"];
#[crate_type = "dylib"];
#[license = "MIT/ASL2"];
#[allow(missing_doc)];
#[forbid(non_camel_case_types)];
#[feature(macro_rules,managed_boxes)];

// test harness access
#[cfg(test)]
extern crate test;

extern crate collections;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable,
                          DecoderHelpers, EncoderHelpers};

mod serialize;
mod collection_impls;

pub mod base64;
pub mod ebml;
pub mod hex;
pub mod json;
