// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The Rust compiler.

# Note

This API is completely unstable and subject to change.

*/

#![crate_name = "rustc"]
#![experimental]
#![comment = "The Rust compiler"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/master/")]

#![allow(deprecated)]
#![allow(unknown_features)]
#![feature(macro_rules, globs, struct_variant, quote)]
#![feature(default_type_params, phase, unsafe_destructor, slicing_syntax)]

#![feature(rustc_diagnostic_macros)]
#![feature(import_shadowing)]

extern crate arena;
extern crate debug;
extern crate flate;
extern crate getopts;
extern crate graphviz;
extern crate libc;
extern crate rustc_llvm;
extern crate rustc_back;
extern crate serialize;
extern crate rbml;
extern crate time;
#[phase(plugin, link)] extern crate log;
#[phase(plugin, link)] extern crate syntax;

#[cfg(test)]
extern crate test;

pub use rustc_llvm as llvm;

mod diagnostics;

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

    pub mod link;
    pub mod lto;
    pub mod write;

}

pub mod middle {
    pub mod astencode;
    pub mod borrowck;
    pub mod cfg;
    pub mod check_const;
    pub mod check_static_recursion;
    pub mod check_loop;
    pub mod check_match;
    pub mod check_rvalues;
    pub mod check_static;
    pub mod const_eval;
    pub mod dataflow;
    pub mod dead;
    pub mod def;
    pub mod dependency_format;
    pub mod effect;
    pub mod entry;
    pub mod expr_use_visitor;
    pub mod graph;
    pub mod intrinsicck;
    pub mod lang_items;
    pub mod liveness;
    pub mod mem_categorization;
    pub mod pat_util;
    pub mod privacy;
    pub mod reachable;
    pub mod region;
    pub mod resolve;
    pub mod resolve_lifetime;
    pub mod save;
    pub mod stability;
    pub mod subst;
    pub mod traits;
    pub mod trans;
    pub mod ty;
    pub mod ty_fold;
    pub mod typeck;
    pub mod weak_lang_items;
}

pub mod metadata;

pub mod driver;

pub mod plugin;

pub mod lint;

pub mod util {
    pub use rustc_back::fs;
    pub use rustc_back::sha2;

    pub mod common;
    pub mod ppaux;
    pub mod nodemap;
    pub mod snapshot_vec;
}

pub mod lib {
    pub use llvm;
}

__build_diagnostic_array!(DIAGNOSTICS)

// A private module so that macro-expanded idents like
// `::rustc::lint::Lint` will also work in `rustc` itself.
//
// `libstd` uses the same trick.
#[doc(hidden)]
mod rustc {
    pub use lint;
}

pub fn main() {
    let args = std::os::args();
    std::os::set_exit_status(driver::main_args(args.as_slice()));
}
