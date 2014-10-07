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
#![experimental]
#![comment = "The Rust compiler minimal-dependency dumping-ground"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/")]

#![allow(unknown_features)]
#![feature(globs, phase, macro_rules, slicing_syntax)]

#[phase(plugin, link)]
extern crate log;
extern crate syntax;
extern crate serialize;

pub mod abi;
pub mod archive;
pub mod arm;
pub mod fs;
pub mod mips;
pub mod mipsel;
pub mod rpath;
pub mod sha2;
pub mod svh;
pub mod target_strs;
pub mod x86;
pub mod x86_64;
