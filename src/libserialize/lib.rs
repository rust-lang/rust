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
#[doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://static.rust-lang.org/doc/master")];
#[allow(missing_doc)];
#[forbid(non_camel_case_types)];
#[feature(macro_rules, managed_boxes, default_type_params, phase)];

// NOTE remove the following two attributes after the next snapshot.
#[allow(unrecognized_lint)];
#[allow(default_type_param_usage)];
#[allow(deprecated_owned_vector)];

// test harness access
#[cfg(test)]
extern crate test;
#[phase(syntax, link)]
extern crate log;

extern crate collections;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable,
                          DecoderHelpers, EncoderHelpers};

mod serialize;
mod collection_impls;

pub mod base64;
pub mod ebml;
pub mod hex;
pub mod json;
