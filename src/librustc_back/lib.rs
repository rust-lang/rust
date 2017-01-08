// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some stuff used by rustc that doesn't have many dependencies
//!
//! Originally extracted from rustc::back, which was nominally the
//! compiler 'backend', though LLVM is rustc's backend, so rustc_back
//! is really just odds-and-ends relating to code gen and linking.
//! This crate mostly exists to make rustc smaller, so we might put
//! more 'stuff' here in the future.  It does not have a dependency on
//! rustc_llvm.
//!
//! FIXME: Split this into two crates: one that has deps on syntax, and
//! one that doesn't; the one that doesn't might get decent parallel
//! build speedups.

#![crate_name = "rustc_back"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(libc)]
#![feature(rand)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![cfg_attr(test, feature(rand))]

extern crate syntax;
extern crate libc;
extern crate serialize;
#[macro_use] extern crate log;

extern crate serialize as rustc_serialize; // used by deriving

pub mod tempdir;
pub mod target;
pub mod slice;
pub mod dynamic_lib;

use serialize::json::{Json, ToJson};

#[derive(Clone, Copy, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub enum PanicStrategy {
    Unwind,
    Abort,
}

impl PanicStrategy {
    pub fn desc(&self) -> &str {
        match *self {
            PanicStrategy::Unwind => "unwind",
            PanicStrategy::Abort => "abort",
        }
    }
}

impl ToJson for PanicStrategy {
    fn to_json(&self) -> Json {
        match *self {
            PanicStrategy::Abort => "abort".to_json(),
            PanicStrategy::Unwind => "unwind".to_json(),
        }
    }
}
