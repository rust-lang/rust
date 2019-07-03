#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(core_intrinsics)]
#![feature(libc)]
#![feature(rustc_diagnostic_macros)]
#![feature(stmt_expr_attributes)]
#![feature(try_blocks)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(trusted_len)]
#![feature(mem_take)]
#![allow(unused_attributes)]
#![allow(dead_code)]
#![deny(rust_2018_idioms)]
#![deny(internal)]
#![deny(unused_lifetimes)]

#![recursion_limit="256"]

//! This crate contains codegen code that is used by all codegen backends (LLVM and others).
//! The backend-agnostic functions of this crate use functions defined in various traits that
//! have to be implemented by each backends.

#[macro_use] extern crate log;
#[macro_use] extern crate rustc;
#[macro_use] extern crate rustc_data_structures;
#[macro_use] extern crate syntax;

use std::path::PathBuf;
use rustc::dep_graph::WorkProduct;
use rustc::session::config::{OutputFilenames, OutputType};
use rustc::middle::lang_items::LangItem;
use rustc::hir::def_id::CrateNum;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::svh::Svh;
use rustc::middle::cstore::{LibSource, CrateSource, NativeLibrary};
use syntax_pos::symbol::Symbol;

// N.B., this module needs to be declared first so diagnostics are
// registered before they are used.
mod error_codes;

pub mod common;
pub mod traits;
pub mod mir;
pub mod debuginfo;
pub mod base;
pub mod callee;
pub mod glue;
pub mod meth;
pub mod mono_item;
pub mod back;

pub struct ModuleCodegen<M> {
    /// The name of the module. When the crate may be saved between
    /// compilations, incremental compilation requires that name be
    /// unique amongst **all** crates. Therefore, it should contain
    /// something unique to this crate (e.g., a module path) as well
    /// as the crate name and disambiguator.
    /// We currently generate these names via CodegenUnit::build_cgu_name().
    pub name: String,
    pub module_llvm: M,
    pub kind: ModuleKind,
}

pub const METADATA_FILENAME: &str = "rust.metadata.bin";
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

bitflags::bitflags! {
    pub struct MemFlags: u8 {
        const VOLATILE = 1 << 0;
        const NONTEMPORAL = 1 << 1;
        const UNALIGNED = 1 << 2;
    }
}

/// Misc info we load from metadata to persist beyond the tcx.
pub struct CrateInfo {
    pub panic_runtime: Option<CrateNum>,
    pub compiler_builtins: Option<CrateNum>,
    pub profiler_runtime: Option<CrateNum>,
    pub sanitizer_runtime: Option<CrateNum>,
    pub is_no_builtins: FxHashSet<CrateNum>,
    pub native_libraries: FxHashMap<CrateNum, Lrc<Vec<NativeLibrary>>>,
    pub crate_name: FxHashMap<CrateNum, String>,
    pub used_libraries: Lrc<Vec<NativeLibrary>>,
    pub link_args: Lrc<Vec<String>>,
    pub used_crate_source: FxHashMap<CrateNum, Lrc<CrateSource>>,
    pub used_crates_static: Vec<(CrateNum, LibSource)>,
    pub used_crates_dynamic: Vec<(CrateNum, LibSource)>,
    pub lang_item_to_crate: FxHashMap<LangItem, CrateNum>,
    pub missing_lang_items: FxHashMap<CrateNum, Vec<LangItem>>,
}


pub struct CodegenResults {
    pub crate_name: Symbol,
    pub modules: Vec<CompiledModule>,
    pub allocator_module: Option<CompiledModule>,
    pub metadata_module: Option<CompiledModule>,
    pub crate_hash: Svh,
    pub metadata: rustc::middle::cstore::EncodedMetadata,
    pub windows_subsystem: Option<String>,
    pub linker_info: back::linker::LinkerInfo,
    pub crate_info: CrateInfo,
}

__build_diagnostic_array! { librustc_codegen_ssa, DIAGNOSTICS }
