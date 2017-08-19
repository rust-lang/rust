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
#![crate_type = "dylib"]
#![crate_type = "rlib"]
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
#![feature(libc)]
#![feature(quote)]
#![feature(rustc_diagnostic_macros)]
#![feature(slice_patterns)]
#![feature(conservative_impl_trait)]

use rustc::dep_graph::WorkProduct;
use syntax_pos::symbol::Symbol;

extern crate flate2;
extern crate libc;
extern crate owning_ref;
#[macro_use] extern crate rustc;
extern crate rustc_allocator;
extern crate rustc_back;
extern crate rustc_data_structures;
extern crate rustc_incremental;
extern crate rustc_llvm as llvm;
extern crate rustc_platform_intrinsics as intrinsics;
extern crate rustc_const_math;
#[macro_use]
#[no_link]
extern crate rustc_bitflags;
extern crate rustc_demangle;
extern crate jobserver;
extern crate num_cpus;

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate rustc_errors as errors;
extern crate serialize;
#[cfg(windows)]
extern crate gcc; // Used to locate MSVC, not gcc :)

pub use base::trans_crate;
pub use back::symbol_names::provide;

pub use metadata::LlvmMetadataLoader;
pub use llvm_util::{init, target_features, print_version, print_passes, print, enable_llvm_debug};

pub mod back {
    mod archive;
    pub(crate) mod linker;
    pub mod link;
    mod lto;
    pub(crate) mod symbol_export;
    pub(crate) mod symbol_names;
    pub mod write;
    mod rpath;
}

mod diagnostics;

mod abi;
mod adt;
mod allocator;
mod asm;
mod assert_module_sources;
mod attributes;
mod base;
mod builder;
mod cabi_aarch64;
mod cabi_arm;
mod cabi_asmjs;
mod cabi_hexagon;
mod cabi_mips;
mod cabi_mips64;
mod cabi_msp430;
mod cabi_nvptx;
mod cabi_nvptx64;
mod cabi_powerpc;
mod cabi_powerpc64;
mod cabi_s390x;
mod cabi_sparc;
mod cabi_sparc64;
mod cabi_x86;
mod cabi_x86_64;
mod cabi_x86_win64;
mod callee;
mod collector;
mod common;
mod consts;
mod context;
mod debuginfo;
mod declare;
mod glue;
mod intrinsic;
mod llvm_util;
mod machine;
mod metadata;
mod meth;
mod mir;
mod monomorphize;
mod partitioning;
mod symbol_names_test;
mod time_graph;
mod trans_item;
mod tvec;
mod type_;
mod type_of;
mod value;

pub struct ModuleTranslation {
    /// The name of the module. When the crate may be saved between
    /// compilations, incremental compilation requires that name be
    /// unique amongst **all** crates.  Therefore, it should contain
    /// something unique to this crate (e.g., a module path) as well
    /// as the crate name and disambiguator.
    name: String,
    symbol_name_hash: u64,
    pub source: ModuleSource,
    pub kind: ModuleKind,
}

#[derive(Copy, Clone, Debug)]
pub enum ModuleKind {
    Regular,
    Metadata,
    Allocator,
}

impl ModuleTranslation {
    pub fn into_compiled_module(self, emit_obj: bool, emit_bc: bool) -> CompiledModule {
        let pre_existing = match self.source {
            ModuleSource::Preexisting(_) => true,
            ModuleSource::Translated(_) => false,
        };

        CompiledModule {
            name: self.name.clone(),
            kind: self.kind,
            symbol_name_hash: self.symbol_name_hash,
            pre_existing,
            emit_obj,
            emit_bc,
        }
    }
}

impl Drop for ModuleTranslation {
    fn drop(&mut self) {
        match self.source {
            ModuleSource::Preexisting(_) => {
                // Nothing to dispose.
            },
            ModuleSource::Translated(llvm) => {
                unsafe {
                    llvm::LLVMDisposeModule(llvm.llmod);
                    llvm::LLVMContextDispose(llvm.llcx);
                }
            },
        }
    }
}

#[derive(Debug)]
pub struct CompiledModule {
    pub name: String,
    pub kind: ModuleKind,
    pub symbol_name_hash: u64,
    pub pre_existing: bool,
    pub emit_obj: bool,
    pub emit_bc: bool,
}

#[derive(Clone)]
pub enum ModuleSource {
    /// Copy the `.o` files or whatever from the incr. comp. directory.
    Preexisting(WorkProduct),

    /// Rebuild from this LLVM module.
    Translated(ModuleLlvm),
}

#[derive(Copy, Clone, Debug)]
pub struct ModuleLlvm {
    llcx: llvm::ContextRef,
    pub llmod: llvm::ModuleRef,
}

unsafe impl Send for ModuleTranslation { }
unsafe impl Sync for ModuleTranslation { }

pub struct CrateTranslation {
    pub crate_name: Symbol,
    pub modules: Vec<CompiledModule>,
    allocator_module: Option<CompiledModule>,
    pub link: rustc::middle::cstore::LinkMeta,
    pub metadata: rustc::middle::cstore::EncodedMetadata,
    windows_subsystem: Option<String>,
    linker_info: back::linker::LinkerInfo
}

__build_diagnostic_array! { librustc_trans, DIAGNOSTICS }
