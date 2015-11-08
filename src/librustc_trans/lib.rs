// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The Rust compiler.
//!
//! # Note
//!
//! This API is completely unstable and subject to change.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "rustc_trans"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![staged_api]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(iter_cmp)]
#![feature(iter_arith)]
#![feature(libc)]
#![feature(path_ext)]
#![feature(path_ext)]
#![feature(path_relative_from)]
#![feature(path_relative_from)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(unicode)]
#![feature(unicode)]
#![feature(vec_push_all)]

#![allow(trivial_casts)]

extern crate arena;
extern crate flate;
extern crate getopts;
extern crate graphviz;
extern crate libc;
extern crate rustc;
extern crate rustc_back;
extern crate rustc_front;
extern crate rustc_llvm as llvm;
extern crate rustc_platform_intrinsics as intrinsics;
extern crate serialize;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;

pub use rustc::session;
pub use rustc::metadata;
pub use rustc::middle;
pub use rustc::lint;
pub use rustc::plugin;
pub use rustc::util;

pub mod back {
    pub use rustc_back::abi;
    pub use rustc_back::rpath;
    pub use rustc_back::svh;

    pub mod archive;
    pub mod linker;
    pub mod link;
    pub mod lto;
    pub mod write;
    pub mod msvc;
}

pub mod diagnostics;

pub mod trans;
pub mod save;

pub mod lib {
    pub use llvm;
}
