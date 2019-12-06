#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

#![feature(bool_to_option)]
#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(core_intrinsics)]
#![feature(libc)]
#![feature(slice_patterns)]
#![feature(stmt_expr_attributes)]
#![feature(try_blocks)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(trusted_len)]
#![feature(associated_type_bounds)]

#![recursion_limit="256"]

//! This crate contains codegen code that is used by all codegen backends (LLVM and others).
//! The backend-agnostic functions of this crate use functions defined in various traits that
//! have to be implemented by each backends.

#[macro_use] extern crate log;
#[macro_use] extern crate rustc;
#[macro_use] extern crate syntax;

use std::path::{Path, PathBuf};
use rustc::dep_graph::WorkProduct;
use rustc::session::config::{OutputFilenames, OutputType, RUST_CGU_EXT};
use rustc::middle::lang_items::LangItem;
use rustc::hir::def_id::CrateNum;
use rustc::ty::query::Providers;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::svh::Svh;
use rustc::middle::cstore::{LibSource, CrateSource, NativeLibrary};
use rustc::middle::dependency_format::Dependencies;
use syntax_pos::symbol::Symbol;

pub mod common;
pub mod traits;
pub mod mir;
pub mod debuginfo;
pub mod base;
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

// FIXME(eddyb) maybe include the crate name in this?
pub const METADATA_FILENAME: &str = "lib.rmeta";
pub const RLIB_BYTECODE_EXTENSION: &str = "bc.z";


impl<M> ModuleCodegen<M> {
    pub fn into_compiled_module(self,
                            emit_obj: bool,
                            emit_bc: bool,
                            emit_bc_compressed: bool,
                            outputs: &OutputFilenames) -> CompiledModule {
        let object = emit_obj
            .then(|| outputs.temp_path(OutputType::Object, Some(&self.name)));
        let bytecode = emit_bc
            .then(|| outputs.temp_path(OutputType::Bitcode, Some(&self.name)));
        let bytecode_compressed = emit_bc_compressed.then(|| {
            outputs.temp_path(OutputType::Bitcode, Some(&self.name))
                .with_extension(RLIB_BYTECODE_EXTENSION)
        });

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
#[derive(Debug)]
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
    pub dependency_formats: Lrc<Dependencies>,
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

pub fn provide(providers: &mut Providers<'_>) {
    crate::back::symbol_export::provide(providers);
    crate::base::provide_both(providers);
}

pub fn provide_extern(providers: &mut Providers<'_>) {
    crate::back::symbol_export::provide_extern(providers);
    crate::base::provide_both(providers);
}

/// Checks if the given filename ends with the `.rcgu.o` extension that `rustc`
/// uses for the object files it generates.
pub fn looks_like_rust_object_file(filename: &str) -> bool {
    let path = Path::new(filename);
    let ext = path.extension().and_then(|s| s.to_str());
    if ext != Some(OutputType::Object.extension()) {
        // The file name does not end with ".o", so it can't be an object file.
        return false
    }

    // Strip the ".o" at the end
    let ext2 = path.file_stem()
        .and_then(|s| Path::new(s).extension())
        .and_then(|s| s.to_str());

    // Check if the "inner" extension
    ext2 == Some(RUST_CGU_EXT)
}
