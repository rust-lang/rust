// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "rustc_metadata"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![cfg_attr(stage0, staged_api)]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(enumset)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(time2)]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
#[macro_use] #[no_link] extern crate rustc_bitflags;

extern crate flate;
extern crate rbml;
extern crate serialize;

extern crate rustc;
extern crate rustc_back;
extern crate rustc_front;
extern crate rustc_llvm;

pub use rustc::middle;

#[macro_use]
mod macros;

pub mod diagnostics;

pub mod astencode;
pub mod common;
pub mod tyencode;
pub mod tydecode;
pub mod encoder;
pub mod decoder;
pub mod creader;
pub mod csearch;
pub mod cstore;
pub mod index;
pub mod loader;
pub mod macro_import;
pub mod tls_context;
