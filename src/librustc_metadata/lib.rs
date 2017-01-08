// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_metadata"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_patterns)]
#![feature(conservative_impl_trait)]
#![feature(core_intrinsics)]
#![feature(proc_macro_internals)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(specialization)]
#![feature(staged_api)]

#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate flate;
extern crate serialize as rustc_serialize; // used by deriving
extern crate rustc_errors as errors;
extern crate syntax_ext;
extern crate proc_macro;

#[macro_use]
extern crate rustc;
extern crate rustc_back;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate rustc_llvm;
extern crate rustc_i128;

mod diagnostics;

pub use rustc::middle;

mod astencode;
mod index_builder;
mod index;
mod encoder;
mod decoder;
mod cstore_impl;
mod schema;

pub mod creader;
pub mod cstore;
pub mod locator;

__build_diagnostic_array! { librustc_metadata, DIAGNOSTICS }
