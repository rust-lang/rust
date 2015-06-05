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
//!
//! # Usage
//!
//! This crate is [on crates.io](https://crates.io/crates/rustc-serialize) and
//! can be used by adding `rustc-serialize` to the dependencies in your
//! project's `Cargo.toml`.
//!
//! ```toml
//! [dependencies]
//! rustc-serialize = "0.3"
//! ```
//!
//! and this to your crate root:
//!
//! ```rust
//! extern crate rustc_serialize;
//! ```

#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/rustc-serialize/")]
#![cfg_attr(test, deny(warnings))]
#![allow(trivial_numeric_casts)]
#![cfg_attr(rust_build, feature(staged_api))]
#![cfg_attr(rust_build, staged_api)]
#![cfg_attr(rust_build,
            unstable(feature = "rustc_private",
                     reason = "use the crates.io `rustc-serialize` library instead"))]

#[cfg(test)] extern crate rand;

pub use self::serialize::{Decoder, Encoder, Decodable, Encodable,
                          DecoderHelpers, EncoderHelpers};

mod serialize;
mod collection_impls;

pub mod base64;
pub mod hex;
pub mod json;

mod rustc_serialize {
    pub use serialize::*;
}
