// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(file_buffered)]
#![feature(if_let_guard)]
#![feature(negative_impls)]
#![feature(rustdoc_internals)]
#![feature(string_from_utf8_lossy_owned)]
#![feature(trait_alias)]
#![feature(try_blocks)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

//! This crate contains codegen code that is used by all codegen backends (LLVM and others).
//! The backend-agnostic functions of this crate use functions defined in various traits that
//! have to be implemented by each backend.

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_data_structures::unord::UnordMap;
use rustc_hir::CRATE_HIR_ID;
use rustc_hir::def_id::CrateNum;
use rustc_macros::{Decodable, Encodable, HashStable};
use rustc_middle::dep_graph::WorkProduct;
use rustc_middle::lint::LevelAndSource;
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::middle::dependency_format::Dependencies;
use rustc_middle::middle::exported_symbols::SymbolExportKind;
use rustc_middle::ty::TyCtxt;
use rustc_middle::util::Providers;
use rustc_serialize::opaque::{FileEncoder, MemDecoder};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_session::Session;
use rustc_session::config::{CrateType, OutputFilenames, OutputType, RUST_CGU_EXT};
use rustc_session::cstore::{self, CrateSource};
use rustc_session::lint::builtin::LINKER_MESSAGES;
use rustc_session::utils::NativeLibKind;
use rustc_span::Symbol;

pub mod assert_module_sources;
pub mod back;
pub mod base;
pub mod codegen_attrs;
pub mod common;
pub mod debuginfo;
pub mod errors;
pub mod meth;
pub mod mir;
pub mod mono_item;
pub mod size_of_val;
pub mod target_features;
pub mod traits;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

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
    /// Saving the ThinLTO buffer for embedding in the object file.
    pub thin_lto_buffer: Option<Vec<u8>>,
}

impl<M> ModuleCodegen<M> {
    pub fn new_regular(name: impl Into<String>, module: M) -> Self {
        Self {
            name: name.into(),
            module_llvm: module,
            kind: ModuleKind::Regular,
            thin_lto_buffer: None,
        }
    }

    pub fn new_allocator(name: impl Into<String>, module: M) -> Self {
        Self {
            name: name.into(),
            module_llvm: module,
            kind: ModuleKind::Allocator,
            thin_lto_buffer: None,
        }
    }

    pub fn into_compiled_module(
        self,
        emit_obj: bool,
        emit_dwarf_obj: bool,
        emit_bc: bool,
        emit_asm: bool,
        emit_ir: bool,
        outputs: &OutputFilenames,
        invocation_temp: Option<&str>,
    ) -> CompiledModule {
        let object = emit_obj
            .then(|| outputs.temp_path_for_cgu(OutputType::Object, &self.name, invocation_temp));
        let dwarf_object =
            emit_dwarf_obj.then(|| outputs.temp_path_dwo_for_cgu(&self.name, invocation_temp));
        let bytecode = emit_bc
            .then(|| outputs.temp_path_for_cgu(OutputType::Bitcode, &self.name, invocation_temp));
        let assembly = emit_asm
            .then(|| outputs.temp_path_for_cgu(OutputType::Assembly, &self.name, invocation_temp));
        let llvm_ir = emit_ir.then(|| {
            outputs.temp_path_for_cgu(OutputType::LlvmAssembly, &self.name, invocation_temp)
        });

        CompiledModule {
            name: self.name.clone(),
            kind: self.kind,
            object,
            dwarf_object,
            bytecode,
            assembly,
            llvm_ir,
            links_from_incr_cache: Vec::new(),
        }
    }
}

#[derive(Debug, Encodable, Decodable)]
pub struct CompiledModule {
    pub name: String,
    pub kind: ModuleKind,
    pub object: Option<PathBuf>,
    pub dwarf_object: Option<PathBuf>,
    pub bytecode: Option<PathBuf>,
    pub assembly: Option<PathBuf>, // --emit=asm
    pub llvm_ir: Option<PathBuf>,  // --emit=llvm-ir, llvm-bc is in bytecode
    pub links_from_incr_cache: Vec<PathBuf>,
}

impl CompiledModule {
    /// Call `emit` function with every artifact type currently compiled
    pub fn for_each_output(&self, mut emit: impl FnMut(&Path, OutputType)) {
        if let Some(path) = self.object.as_deref() {
            emit(path, OutputType::Object);
        }
        if let Some(path) = self.bytecode.as_deref() {
            emit(path, OutputType::Bitcode);
        }
        if let Some(path) = self.llvm_ir.as_deref() {
            emit(path, OutputType::LlvmAssembly);
        }
        if let Some(path) = self.assembly.as_deref() {
            emit(path, OutputType::Assembly);
        }
    }
}

pub(crate) struct CachedModuleCodegen {
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
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MemFlags: u8 {
        const VOLATILE = 1 << 0;
        const NONTEMPORAL = 1 << 1;
        const UNALIGNED = 1 << 2;
    }
}

#[derive(Clone, Debug, Encodable, Decodable, HashStable)]
pub struct NativeLib {
    pub kind: NativeLibKind,
    pub name: Symbol,
    pub filename: Option<Symbol>,
    pub cfg: Option<ast::MetaItemInner>,
    pub verbatim: bool,
    pub dll_imports: Vec<cstore::DllImport>,
}

impl From<&cstore::NativeLib> for NativeLib {
    fn from(lib: &cstore::NativeLib) -> Self {
        NativeLib {
            kind: lib.kind,
            filename: lib.filename,
            name: lib.name,
            cfg: lib.cfg.clone(),
            verbatim: lib.verbatim.unwrap_or(false),
            dll_imports: lib.dll_imports.clone(),
        }
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
    pub target_cpu: String,
    pub target_features: Vec<String>,
    pub crate_types: Vec<CrateType>,
    pub exported_symbols: UnordMap<CrateType, Vec<(String, SymbolExportKind)>>,
    pub linked_symbols: FxIndexMap<CrateType, Vec<(String, SymbolExportKind)>>,
    pub local_crate_name: Symbol,
    pub compiler_builtins: Option<CrateNum>,
    pub profiler_runtime: Option<CrateNum>,
    pub is_no_builtins: FxHashSet<CrateNum>,
    pub native_libraries: FxIndexMap<CrateNum, Vec<NativeLib>>,
    pub crate_name: UnordMap<CrateNum, Symbol>,
    pub used_libraries: Vec<NativeLib>,
    pub used_crate_source: UnordMap<CrateNum, Arc<CrateSource>>,
    pub used_crates: Vec<CrateNum>,
    pub dependency_formats: Arc<Dependencies>,
    pub windows_subsystem: Option<String>,
    pub natvis_debugger_visualizers: BTreeSet<DebuggerVisualizerFile>,
    pub lint_levels: CodegenLintLevels,
}

/// Target-specific options that get set in `cfg(...)`.
///
/// RUSTC_SPECIFIC_FEATURES should be skipped here, those are handled outside codegen.
pub struct TargetConfig {
    /// Options to be set in `cfg(target_features)`.
    pub target_features: Vec<Symbol>,
    /// Options to be set in `cfg(target_features)`, but including unstable features.
    pub unstable_target_features: Vec<Symbol>,
    /// Option for `cfg(target_has_reliable_f16)`, true if `f16` basic arithmetic works.
    pub has_reliable_f16: bool,
    /// Option for `cfg(target_has_reliable_f16_math)`, true if `f16` math calls work.
    pub has_reliable_f16_math: bool,
    /// Option for `cfg(target_has_reliable_f128)`, true if `f128` basic arithmetic works.
    pub has_reliable_f128: bool,
    /// Option for `cfg(target_has_reliable_f128_math)`, true if `f128` math calls work.
    pub has_reliable_f128_math: bool,
}

#[derive(Encodable, Decodable)]
pub struct CodegenResults {
    pub modules: Vec<CompiledModule>,
    pub allocator_module: Option<CompiledModule>,
    pub metadata_module: Option<CompiledModule>,
    pub metadata: rustc_metadata::EncodedMetadata,
    pub crate_info: CrateInfo,
}

pub enum CodegenErrors {
    WrongFileType,
    EmptyVersionNumber,
    EncodingVersionMismatch { version_array: String, rlink_version: u32 },
    RustcVersionMismatch { rustc_version: String },
    CorruptFile,
}

pub fn provide(providers: &mut Providers) {
    crate::back::symbol_export::provide(providers);
    crate::base::provide(providers);
    crate::target_features::provide(providers);
    crate::codegen_attrs::provide(providers);
    providers.queries.global_backend_features = |_tcx: TyCtxt<'_>, ()| vec![];
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

const RLINK_VERSION: u32 = 1;
const RLINK_MAGIC: &[u8] = b"rustlink";

impl CodegenResults {
    pub fn serialize_rlink(
        sess: &Session,
        rlink_file: &Path,
        codegen_results: &CodegenResults,
        outputs: &OutputFilenames,
    ) -> Result<usize, io::Error> {
        let mut encoder = FileEncoder::new(rlink_file)?;
        encoder.emit_raw_bytes(RLINK_MAGIC);
        // `emit_raw_bytes` is used to make sure that the version representation does not depend on
        // Encoder's inner representation of `u32`.
        encoder.emit_raw_bytes(&RLINK_VERSION.to_be_bytes());
        encoder.emit_str(sess.cfg_version);
        Encodable::encode(codegen_results, &mut encoder);
        Encodable::encode(outputs, &mut encoder);
        encoder.finish().map_err(|(_path, err)| err)
    }

    pub fn deserialize_rlink(
        sess: &Session,
        data: Vec<u8>,
    ) -> Result<(Self, OutputFilenames), CodegenErrors> {
        // The Decodable machinery is not used here because it panics if the input data is invalid
        // and because its internal representation may change.
        if !data.starts_with(RLINK_MAGIC) {
            return Err(CodegenErrors::WrongFileType);
        }
        let data = &data[RLINK_MAGIC.len()..];
        if data.len() < 4 {
            return Err(CodegenErrors::EmptyVersionNumber);
        }

        let mut version_array: [u8; 4] = Default::default();
        version_array.copy_from_slice(&data[..4]);
        if u32::from_be_bytes(version_array) != RLINK_VERSION {
            return Err(CodegenErrors::EncodingVersionMismatch {
                version_array: String::from_utf8_lossy(&version_array).to_string(),
                rlink_version: RLINK_VERSION,
            });
        }

        let Ok(mut decoder) = MemDecoder::new(&data[4..], 0) else {
            return Err(CodegenErrors::CorruptFile);
        };
        let rustc_version = decoder.read_str();
        if rustc_version != sess.cfg_version {
            return Err(CodegenErrors::RustcVersionMismatch {
                rustc_version: rustc_version.to_string(),
            });
        }

        let codegen_results = CodegenResults::decode(&mut decoder);
        let outputs = OutputFilenames::decode(&mut decoder);
        Ok((codegen_results, outputs))
    }
}

/// A list of lint levels used in codegen.
///
/// When using `-Z link-only`, we don't have access to the tcx and must work
/// solely from the `.rlink` file. `Lint`s are defined too early to be encodeable.
/// Instead, encode exactly the information we need.
#[derive(Copy, Clone, Debug, Encodable, Decodable)]
pub struct CodegenLintLevels {
    linker_messages: LevelAndSource,
}

impl CodegenLintLevels {
    pub fn from_tcx(tcx: TyCtxt<'_>) -> Self {
        Self { linker_messages: tcx.lint_level_at_node(LINKER_MESSAGES, CRATE_HIR_ID) }
    }
}
