// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "rustc_back#0.11.0-pre"]
#![crate_name = "rustc_back"]
#![experimental]
#![comment = "The Rust compiler backend"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/")]

#![feature(globs)]
#![feature(phase)]
#![allow(unused_attribute)] // NOTE: remove after stage0

#[phase(plugin, link)]
extern crate log;
extern crate syntax;
extern crate libc;
extern crate flate;

pub mod abi;
pub mod archive;
pub mod arm;
pub mod mips;
pub mod mipsel;
pub mod rpath;
pub mod svh;
pub mod target_strs;
pub mod x86;
pub mod x86_64;
