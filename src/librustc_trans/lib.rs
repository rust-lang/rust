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

#![crate_name = "rustc_trans"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
      html_root_url = "https://doc.rust-lang.org/nightly/")]
#![cfg_attr(not(stage0), deny(warnings))]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(const_fn)]
#![feature(custom_attribute)]
#![allow(unused_attributes)]
#![feature(iter_arith)]
#![feature(libc)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(slice_patterns)]
#![feature(staged_api)]
#![feature(unicode)]
#![feature(question_mark)]

extern crate arena;
extern crate flate;
extern crate getopts;
extern crate graphviz;
extern crate libc;
#[macro_use] extern crate rustc;
extern crate rustc_back;
extern crate rustc_data_structures;
extern crate rustc_incremental;
pub extern crate rustc_llvm as llvm;
extern crate rustc_platform_intrinsics as intrinsics;
extern crate serialize;
extern crate rustc_const_math;
extern crate rustc_const_eval;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;

pub use rustc::session;
pub use rustc::middle;
pub use rustc::lint;
pub use rustc::util;

pub use base::trans_crate;
pub use disr::Disr;

pub mod back {
    pub use rustc_back::rpath;
    pub use rustc::hir::svh;

    pub mod archive;
    pub mod linker;
    pub mod link;
    pub mod lto;
    pub mod symbol_names;
    pub mod write;
    pub mod msvc;
}

pub mod diagnostics;

#[macro_use]
mod macros;

mod abi;
mod adt;
mod asm;
mod attributes;
mod base;
mod basic_block;
mod build;
mod builder;
mod cabi_aarch64;
mod cabi_arm;
mod cabi_asmjs;
mod cabi_mips;
mod cabi_powerpc;
mod cabi_powerpc64;
mod cabi_x86;
mod cabi_x86_64;
mod cabi_x86_win64;
mod callee;
mod cleanup;
mod closure;
mod collector;
mod common;
mod consts;
mod context;
mod controlflow;
mod datum;
mod debuginfo;
mod declare;
mod disr;
mod expr;
mod glue;
mod inline;
mod intrinsic;
mod machine;
mod _match;
mod meth;
mod mir;
mod monomorphize;
mod partitioning;
mod symbol_names_test;
mod trans_item;
mod tvec;
mod type_;
mod type_of;
mod value;

#[derive(Copy, Clone)]
pub struct ModuleTranslation {
    pub llcx: llvm::ContextRef,
    pub llmod: llvm::ModuleRef,
}

unsafe impl Send for ModuleTranslation { }
unsafe impl Sync for ModuleTranslation { }

pub struct CrateTranslation {
    pub modules: Vec<ModuleTranslation>,
    pub metadata_module: ModuleTranslation,
    pub link: middle::cstore::LinkMeta,
    pub metadata: Vec<u8>,
    pub reachable: Vec<String>,
    pub no_builtins: bool,
    pub linker_info: back::linker::LinkerInfo
}

__build_diagnostic_array! { librustc_trans, DIAGNOSTICS }
