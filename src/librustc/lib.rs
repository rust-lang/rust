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

#![crate_id = "rustc#0.11.0-pre"]
#![experimental]
#![comment = "The Rust compiler"]
#![license = "MIT/ASL2"]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/")]

#![allow(deprecated)]
#![feature(macro_rules, globs, struct_variant, managed_boxes, quote,
           default_type_params, phase)]

extern crate arena;
extern crate debug;
extern crate flate;
extern crate getopts;
extern crate graphviz;
extern crate libc;
extern crate serialize;
extern crate syntax;
extern crate time;
#[phase(plugin, link)] extern crate log;

pub mod middle {
    pub mod def;
    pub mod trans;
    pub mod ty;
    pub mod ty_fold;
    pub mod subst;
    pub mod resolve;
    pub mod resolve_lifetime;
    pub mod typeck;
    pub mod check_loop;
    pub mod check_match;
    pub mod check_const;
    pub mod check_static;
    pub mod lint;
    pub mod borrowck;
    pub mod dataflow;
    pub mod mem_categorization;
    pub mod liveness;
    pub mod kind;
    pub mod freevars;
    pub mod pat_util;
    pub mod region;
    pub mod const_eval;
    pub mod astencode;
    pub mod lang_items;
    pub mod privacy;
    pub mod entry;
    pub mod effect;
    pub mod reachable;
    pub mod graph;
    pub mod cfg;
    pub mod dead;
    pub mod expr_use_visitor;
    pub mod dependency_format;
    pub mod weak_lang_items;
    pub mod save;
    pub mod intrinsicck;
}

pub mod front {
    pub mod config;
    pub mod test;
    pub mod std_inject;
    pub mod assign_node_ids_and_map;
    pub mod feature_gate;
    pub mod show_span;
}

pub mod back {
    pub mod abi;
    pub mod archive;
    pub mod arm;
    pub mod link;
    pub mod lto;
    pub mod mips;
    pub mod rpath;
    pub mod svh;
    pub mod target_strs;
    pub mod x86;
    pub mod x86_64;
}

pub mod metadata;

pub mod driver;

pub mod plugin;

pub mod util {
    pub mod common;
    pub mod ppaux;
    pub mod sha2;
    pub mod nodemap;
    pub mod fs;
}

pub mod lib {
    pub mod llvm;
    pub mod llvmdeps;
}

pub fn main() {
    let args = std::os::args().iter()
                              .map(|x| x.to_string())
                              .collect::<Vec<_>>();
    std::os::set_exit_status(driver::main_args(args.as_slice()));
}
