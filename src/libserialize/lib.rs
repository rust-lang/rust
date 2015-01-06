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
#![unstable = "deprecated in favor of rustc-serialize on crates.io"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]
#![allow(unknown_features)]
#![feature(macro_rules, default_type_params, phase, slicing_syntax, globs)]
#![feature(unboxed_closures)]
#![feature(associated_types)]

// test harness access
#[cfg(test)]
extern crate test;

#[cfg(stage0)]
#[phase(plugin, link)]
extern crate log;

#[cfg(not(stage0))]
#[macro_use]
extern crate log;

extern crate unicode;

extern crate collections;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable,
                          DecoderHelpers, EncoderHelpers};

#[cfg(stage0)]
#[path = "serialize_stage0.rs"]
mod serialize;
#[cfg(not(stage0))]
mod serialize;

#[cfg(stage0)]
#[path = "collection_impls_stage0.rs"]
mod collection_impls;
#[cfg(not(stage0))]
mod collection_impls;

pub mod base64;
pub mod hex;

#[cfg(stage0)]
#[path = "json_stage0.rs"]
pub mod json;
#[cfg(not(stage0))]
pub mod json;

mod rustc_serialize {
    pub use serialize::*;
}
