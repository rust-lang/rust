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

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(custom_attribute)]
#![feature(libc)]
#![feature(rustc_diagnostic_macros)]
#![feature(in_band_lifetimes)]
#![feature(slice_sort_by_cached_key)]
#![feature(nll)]
#![allow(unused_attributes)]
#![allow(dead_code)]
#![feature(quote)]

#[macro_use] extern crate bitflags;
#[macro_use] extern crate log;
extern crate rustc_apfloat;
#[macro_use]  extern crate rustc;
extern crate rustc_target;
extern crate rustc_mir;
#[macro_use] extern crate syntax;
extern crate syntax_pos;
extern crate rustc_incremental;
extern crate rustc_codegen_utils;
extern crate rustc_data_structures;
extern crate libc;

use std::path::PathBuf;
use rustc::dep_graph::WorkProduct;
use rustc::session::config::{OutputFilenames, OutputType};
use rustc::middle::lang_items::LangItem;
use rustc::hir::def_id::CrateNum;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc::middle::cstore::{LibSource, CrateSource, NativeLibrary};

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
mod diagnostics;

pub mod common;
pub mod interfaces;
pub mod mir;
pub mod debuginfo;
pub mod base;
pub mod callee;
pub mod glue;
pub mod meth;
pub mod mono_item;

pub struct ModuleCodegen<M> {
    /// The name of the module. When the crate may be saved between
    /// compilations, incremental compilation requires that name be
    /// unique amongst **all** crates.  Therefore, it should contain
    /// something unique to this crate (e.g., a module path) as well
    /// as the crate name and disambiguator.
    /// We currently generate these names via CodegenUnit::build_cgu_name().
    pub name: String,
    pub module_llvm: M,
    pub kind: ModuleKind,
}

pub const RLIB_BYTECODE_EXTENSION: &str = "bc.z";

impl<M> ModuleCodegen<M> {
    pub fn into_compiled_module(self,
                            emit_obj: bool,
                            emit_bc: bool,
                            emit_bc_compressed: bool,
                            outputs: &OutputFilenames) -> CompiledModule {
        let object = if emit_obj {
            Some(outputs.temp_path(OutputType::Object, Some(&self.name)))
        } else {
            None
        };
        let bytecode = if emit_bc {
            Some(outputs.temp_path(OutputType::Bitcode, Some(&self.name)))
        } else {
            None
        };
        let bytecode_compressed = if emit_bc_compressed {
            Some(outputs.temp_path(OutputType::Bitcode, Some(&self.name))
                    .with_extension(RLIB_BYTECODE_EXTENSION))
        } else {
            None
        };

        CompiledModule {
            name: self.name.clone(),
            kind: self.kind,
            object,
            bytecode,
            bytecode_compressed,
        }
    }
}

#[derive(Debug)]
pub struct CompiledModule {
    pub name: String,
    pub kind: ModuleKind,
    pub object: Option<PathBuf>,
    pub bytecode: Option<PathBuf>,
    pub bytecode_compressed: Option<PathBuf>,
}

pub struct CachedModuleCodegen {
    pub name: String,
    pub source: WorkProduct,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ModuleKind {
    Regular,
    Metadata,
    Allocator,
}

bitflags! {
    pub struct MemFlags: u8 {
        const VOLATILE = 1 << 0;
        const NONTEMPORAL = 1 << 1;
        const UNALIGNED = 1 << 2;
    }
}

/// Misc info we load from metadata to persist beyond the tcx
struct CrateInfo {
    panic_runtime: Option<CrateNum>,
    compiler_builtins: Option<CrateNum>,
    profiler_runtime: Option<CrateNum>,
    sanitizer_runtime: Option<CrateNum>,
    is_no_builtins: FxHashSet<CrateNum>,
    native_libraries: FxHashMap<CrateNum, Lrc<Vec<NativeLibrary>>>,
    crate_name: FxHashMap<CrateNum, String>,
    used_libraries: Lrc<Vec<NativeLibrary>>,
    link_args: Lrc<Vec<String>>,
    used_crate_source: FxHashMap<CrateNum, Lrc<CrateSource>>,
    used_crates_static: Vec<(CrateNum, LibSource)>,
    used_crates_dynamic: Vec<(CrateNum, LibSource)>,
    wasm_imports: FxHashMap<String, String>,
    lang_item_to_crate: FxHashMap<LangItem, CrateNum>,
    missing_lang_items: FxHashMap<CrateNum, Vec<LangItem>>,
}

__build_diagnostic_array! { librustc_codegen_ssa, DIAGNOSTICS }
