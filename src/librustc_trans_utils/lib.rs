// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Note
//!
//! This API is completely unstable and subject to change.

#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![deny(warnings)]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(i128_type)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_patterns)]
#![feature(conservative_impl_trait)]

extern crate rustc;
extern crate syntax;
extern crate syntax_pos;

pub mod link;
