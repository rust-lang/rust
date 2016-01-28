// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! For now just check_match is moved here. This crate contains
//! things that require librustc and are required by librustc_trans
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

#![crate_name = "rustc_const_eval"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(rustc_diagnostic_macros)]
#![feature(staged_api)]
#![feature(rustc_private)]
#![feature(iter_arith)]

extern crate rustc;
extern crate rustc_front;
extern crate rustc_back;

#[macro_use] extern crate syntax;
#[macro_use] extern crate log;

pub mod diagnostics;

pub mod matches;
