#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![feature(bool_to_option)]
#![feature(option_expect_none)]
#![feature(box_patterns)]
#![feature(drain_filter)]
#![feature(try_blocks)]
#![feature(in_band_lifetimes)]
#![feature(nll)]
#![feature(or_patterns)]
#![feature(associated_type_bounds)]
#![recursion_limit = "256"]

//! This crate contains codegen code that is used by all codegen backends (LLVM and others).
//! The backend-agnostic functions of this crate use functions defined in various traits that
//! have to be implemented by each backends.

#[macro_use]
extern crate rustc_macros;
#[macro_use]
extern crate tracing;
#[macro_use]
extern crate rustc_middle;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_hir::def_id::CrateNum;
use rustc_hir::LangItem;
use rustc_middle::dep_graph::WorkProduct;
use rustc_middle::middle::cstore::{CrateSource, LibSource, NativeLib};
use rustc_middle::middle::dependency_format::Dependencies;
use rustc_middle::ty::query::Providers;
use rustc_session::config::{OutputFilenames, OutputType, RUST_CGU_EXT};
use rustc_span::symbol::Symbol;
use std::path::{Path, PathBuf};

pub mod back;
pub mod base;
pub mod common;
pub mod coverageinfo;
pub mod debuginfo;
pub mod glue;
pub mod meth;
pub mod mir;
pub mod mono_item;
pub mod target_features;
pub mod traits;

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

impl<M> ModuleCodegen<M> {
    pub fn into_compiled_module(
        self,
        emit_obj: bool,
        emit_bc: bool,
        outputs: &OutputFilenames,
    ) -> CompiledModule {
        let object = emit_obj.then(|| outputs.temp_path(OutputType::Object, Some(&self.name)));
        let bytecode = emit_bc.then(|| outputs.temp_path(OutputType::Bitcode, Some(&self.name)));

        CompiledModule { name: self.name.clone(), kind: self.kind, object, bytecode }
    }
}

#[derive(Debug, Encodable, Decodable)]
pub struct CompiledModule {
    pub name: String,
    pub kind: ModuleKind,
    pub object: Option<PathBuf>,
    pub bytecode: Option<PathBuf>,
}

pub struct CachedModuleCodegen {
    pub name: String,
    pub source: WorkProduct,
}

#[derive(Copy, Clone, Debug, PartialEq, Encodable, Decodable)]
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
///
/// Note: though `CrateNum` is only meaningful within the same tcx, information within `CrateInfo`
/// is self-contained. `CrateNum` can be viewed as a unique identifier within a `CrateInfo`, where
/// `used_crate_source` contains all `CrateSource` of the dependents, and maintains a mapping from
/// identifiers (`CrateNum`) to `CrateSource`. The other fields map `CrateNum` to the crate's own
/// additional properties, so that effectively we can retrieve each dependent crate's `CrateSource`
/// and the corresponding properties without referencing information outside of a `CrateInfo`.
#[derive(Debug, Encodable, Decodable)]
pub struct CrateInfo {
    pub panic_runtime: Option<CrateNum>,
    pub compiler_builtins: Option<CrateNum>,
    pub profiler_runtime: Option<CrateNum>,
    pub is_no_builtins: FxHashSet<CrateNum>,
    pub native_libraries: FxHashMap<CrateNum, Lrc<Vec<NativeLib>>>,
    pub crate_name: FxHashMap<CrateNum, String>,
    pub used_libraries: Lrc<Vec<NativeLib>>,
    pub link_args: Lrc<Vec<String>>,
    pub used_crate_source: FxHashMap<CrateNum, Lrc<CrateSource>>,
    pub used_crates_static: Vec<(CrateNum, LibSource)>,
    pub used_crates_dynamic: Vec<(CrateNum, LibSource)>,
    pub lang_item_to_crate: FxHashMap<LangItem, CrateNum>,
    pub missing_lang_items: FxHashMap<CrateNum, Vec<LangItem>>,
    pub dependency_formats: Lrc<Dependencies>,
}

#[derive(Encodable, Decodable)]
pub struct CodegenResults {
    pub crate_name: Symbol,
    pub modules: Vec<CompiledModule>,
    pub allocator_module: Option<CompiledModule>,
    pub metadata_module: Option<CompiledModule>,
    pub metadata: rustc_middle::middle::cstore::EncodedMetadata,
    pub windows_subsystem: Option<String>,
    pub linker_info: back::linker::LinkerInfo,
    pub crate_info: CrateInfo,
}

pub fn provide(providers: &mut Providers) {
    crate::back::symbol_export::provide(providers);
    crate::base::provide_both(providers);
    crate::target_features::provide(providers);
}

pub fn provide_extern(providers: &mut Providers) {
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
        return false;
    }

    // Strip the ".o" at the end
    let ext2 = path.file_stem().and_then(|s| Path::new(s).extension()).and_then(|s| s.to_str());

    // Check if the "inner" extension
    ext2 == Some(RUST_CGU_EXT)
}
