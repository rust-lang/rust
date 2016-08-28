// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Skeleton of RBML (Really Bad Markup Language).
//! See `src/librustc_metadata/reader.rs` for more details.

#![crate_name = "rbml"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       html_playground_url = "https://play.rust-lang.org/",
       test(attr(deny(warnings))))]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(question_mark)]

#![cfg_attr(test, feature(test))]

extern crate serialize;

#[cfg(test)]
extern crate serialize as rustc_serialize; // Used by RustcEncodable

#[macro_use]
extern crate log;

#[cfg(test)]
extern crate test;

pub mod opaque;
pub mod leb128;

pub use self::Error::*;

use std::fmt;

#[derive(Debug)]
pub enum Error {
    IntTooBig(usize),
    InvalidTag(usize),
    Expected(String),
    IoError(std::io::Error),
    ApplicationError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // FIXME: this should be a more useful display form
        fmt::Debug::fmt(self, f)
    }
}
