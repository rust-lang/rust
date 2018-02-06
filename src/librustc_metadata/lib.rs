// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_patterns)]
#![feature(conservative_impl_trait)]
#![feature(fs_read_write)]
#![feature(i128_type)]
#![feature(libc)]
#![feature(proc_macro_internals)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(specialization)]
#![feature(rustc_private)]

extern crate libc;
#[macro_use]
extern crate log;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate flate2;
extern crate serialize as rustc_serialize; // used by deriving
extern crate rustc_errors as errors;
extern crate syntax_ext;
extern crate proc_macro;

#[macro_use]
extern crate rustc;
extern crate rustc_back;
extern crate rustc_data_structures;

mod diagnostics;

mod astencode;
mod index_builder;
mod index;
mod encoder;
mod decoder;
mod cstore_impl;
mod isolated_encoder;
mod schema;
mod native_libs;
mod link_args;

pub mod creader;
pub mod cstore;
pub mod dynamic_lib;
pub mod locator;

#[cfg(not(stage0))] // remove after the next snapshot
__build_diagnostic_array! { librustc_metadata, DIAGNOSTICS }
