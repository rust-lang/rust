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
#![crate_name = "rustc"]
#![unstable(feature = "rustc_private")]
#![staged_api]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(collections)]
#![feature(core)]
#![feature(hash)]
#![feature(libc)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(unsafe_destructor)]
#![feature(staged_api)]
#![feature(std_misc)]
#![feature(path_ext)]
#![feature(str_char)]
#![feature(into_cow)]
#![feature(slice_patterns)]
#![cfg_attr(test, feature(test))]

#![allow(trivial_casts)]

extern crate arena;
extern crate flate;
extern crate fmt_macros;
extern crate getopts;
extern crate graphviz;
extern crate libc;
extern crate rustc_llvm;
extern crate rustc_back;
extern crate rustc_data_structures;
extern crate serialize;
extern crate rbml;
extern crate collections;
#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
#[macro_use] #[no_link] extern crate rustc_bitflags;

extern crate serialize as rustc_serialize; // used by deriving

#[cfg(test)]
extern crate test;

pub use rustc_llvm as llvm;

#[macro_use]
mod macros;

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostics;

pub mod back {
    pub use rustc_back::abi;
    pub use rustc_back::archive;
    pub use rustc_back::arm;
    pub use rustc_back::mips;
    pub use rustc_back::mipsel;
    pub use rustc_back::rpath;
    pub use rustc_back::svh;
    pub use rustc_back::target_strs;
    pub use rustc_back::x86;
    pub use rustc_back::x86_64;
}

pub mod middle {
    pub mod astconv_util;
    pub mod astencode;
    pub mod cfg;
    pub mod check_const;
    pub mod check_static_recursion;
    pub mod check_loop;
    pub mod check_match;
    pub mod check_rvalues;
    pub mod const_eval;
    pub mod dataflow;
    pub mod dead;
    pub mod def;
    pub mod dependency_format;
    pub mod effect;
    pub mod entry;
    pub mod expr_use_visitor;
    pub mod fast_reject;
    pub mod free_region;
    pub mod intrinsicck;
    pub mod infer;
    pub mod implicator;
    pub mod lang_items;
    pub mod liveness;
    pub mod mem_categorization;
    pub mod pat_util;
    pub mod privacy;
    pub mod reachable;
    pub mod region;
    pub mod recursion_limit;
    pub mod resolve_lifetime;
    pub mod stability;
    pub mod subst;
    pub mod traits;
    pub mod ty;
    pub mod ty_fold;
    pub mod ty_match;
    pub mod ty_relate;
    pub mod ty_walk;
    pub mod weak_lang_items;
}

pub mod metadata;

pub mod session;

pub mod plugin;

pub mod lint;

pub mod util {
    pub use rustc_back::fs;
    pub use rustc_back::sha2;

    pub mod common;
    pub mod ppaux;
    pub mod nodemap;
    pub mod lev_distance;
    pub mod num;
}

pub mod lib {
    pub use llvm;
}

// A private module so that macro-expanded idents like
// `::rustc::lint::Lint` will also work in `rustc` itself.
//
// `libstd` uses the same trick.
#[doc(hidden)]
mod rustc {
    pub use lint;
}
