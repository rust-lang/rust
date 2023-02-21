//! the rustc crate store interface. This also includes types that
//! are *mostly* used as a part of that interface, but these should
//! probably get a better home if someone can find one.

use crate::search_paths::PathKind;
use crate::utils::NativeLibKind;
use crate::Session;
use rustc_ast as ast;
use rustc_data_structures::sync::{self, MetadataRef, RwLock};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId, StableCrateId, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash, Definitions};
use rustc_index::vec::IndexVec;
use rustc_span::hygiene::{ExpnHash, ExpnId};
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use rustc_target::spec::Target;

use std::any::Any;
use std::path::{Path, PathBuf};

// lonely orphan structs and enums looking for a better home

/// Where a crate came from on the local filesystem. One of these three options
/// must be non-None.
#[derive(PartialEq, Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct CrateSource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub rmeta: Option<(PathBuf, PathKind)>,
}

impl CrateSource {
    #[inline]
    pub fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.dylib.iter().chain(self.rlib.iter()).chain(self.rmeta.iter()).map(|p| &p.0)
    }
}

#[derive(Encodable, Decodable, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
#[derive(HashStable_Generic)]
pub enum CrateDepKind {
    /// A dependency that is only used for its macros.
    MacrosOnly,
    /// A dependency that is always injected into the dependency list and so
    /// doesn't need to be linked to an rlib, e.g., the injected allocator.
    Implicit,
    /// A dependency that is required by an rlib version of this crate.
    /// Ordinary `extern crate`s result in `Explicit` dependencies.
    Explicit,
}

impl CrateDepKind {
    #[inline]
    pub fn macros_only(self) -> bool {
        match self {
            CrateDepKind::MacrosOnly => true,
            CrateDepKind::Implicit | CrateDepKind::Explicit => false,
        }
    }
}

#[derive(Copy, Debug, PartialEq, Clone, Encodable, Decodable, HashStable_Generic)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

#[derive(Debug, Encodable, Decodable, HashStable_Generic)]
pub struct NativeLib {
    pub kind: NativeLibKind,
    pub name: Option<Symbol>,
    /// If packed_bundled_libs enabled, actual filename of library is stored.
    pub filename: Option<Symbol>,
    pub cfg: Option<ast::MetaItem>,
    pub foreign_module: Option<DefId>,
    pub wasm_import_module: Option<Symbol>,
    pub verbatim: Option<bool>,
    pub dll_imports: Vec<DllImport>,
}

impl NativeLib {
    pub fn has_modifiers(&self) -> bool {
        self.verbatim.is_some() || self.kind.has_modifiers()
    }
}

/// Different ways that the PE Format can decorate a symbol name.
/// From <https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#import-name-type>
#[derive(Copy, Clone, Debug, Encodable, Decodable, HashStable_Generic, PartialEq, Eq)]
pub enum PeImportNameType {
    /// IMPORT_ORDINAL
    /// Uses the ordinal (i.e., a number) rather than the name.
    Ordinal(u16),
    /// Same as IMPORT_NAME
    /// Name is decorated with all prefixes and suffixes.
    Decorated,
    /// Same as IMPORT_NAME_NOPREFIX
    /// Prefix (e.g., the leading `_` or `@`) is skipped, but suffix is kept.
    NoPrefix,
    /// Same as IMPORT_NAME_UNDECORATE
    /// Prefix (e.g., the leading `_` or `@`) and suffix (the first `@` and all
    /// trailing characters) are skipped.
    Undecorated,
}

#[derive(Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub struct DllImport {
    pub name: Symbol,
    pub import_name_type: Option<PeImportNameType>,
    /// Calling convention for the function.
    ///
    /// On x86_64, this is always `DllCallingConvention::C`; on i686, it can be any
    /// of the values, and we use `DllCallingConvention::C` to represent `"cdecl"`.
    pub calling_convention: DllCallingConvention,
    /// Span of import's "extern" declaration; used for diagnostics.
    pub span: Span,
    /// Is this for a function (rather than a static variable).
    pub is_fn: bool,
}

impl DllImport {
    pub fn ordinal(&self) -> Option<u16> {
        if let Some(PeImportNameType::Ordinal(ordinal)) = self.import_name_type {
            Some(ordinal)
        } else {
            None
        }
    }
}

/// Calling convention for a function defined in an external library.
///
/// The usize value, where present, indicates the size of the function's argument list
/// in bytes.
#[derive(Clone, PartialEq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum DllCallingConvention {
    C,
    Stdcall(usize),
    Fastcall(usize),
    Vectorcall(usize),
}

#[derive(Clone, Encodable, Decodable, HashStable_Generic, Debug)]
pub struct ForeignModule {
    pub foreign_items: Vec<DefId>,
    pub def_id: DefId,
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub struct ExternCrate {
    pub src: ExternCrateSource,

    /// span of the extern crate that caused this to be loaded
    pub span: Span,

    /// Number of links to reach the extern;
    /// used to select the extern with the shortest path
    pub path_len: usize,

    /// Crate that depends on this crate
    pub dependency_of: CrateNum,
}

impl ExternCrate {
    /// If true, then this crate is the crate named by the extern
    /// crate referenced above. If false, then this crate is a dep
    /// of the crate.
    #[inline]
    pub fn is_direct(&self) -> bool {
        self.dependency_of == LOCAL_CRATE
    }

    #[inline]
    pub fn rank(&self) -> impl PartialOrd {
        // Prefer:
        // - direct extern crate to indirect
        // - shorter paths to longer
        (self.is_direct(), !self.path_len)
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum ExternCrateSource {
    /// Crate is loaded by `extern crate`.
    Extern(
        /// def_id of the item in the current crate that caused
        /// this crate to be loaded; note that there could be multiple
        /// such ids
        DefId,
    ),
    /// Crate is implicitly loaded by a path resolving through extern prelude.
    Path,
}

/// The backend's way to give the crate store access to the metadata in a library.
/// Note that it returns the raw metadata bytes stored in the library file, whether
/// it is compressed, uncompressed, some weird mix, etc.
/// rmeta files are backend independent and not handled here.
///
/// At the time of this writing, there is only one backend and one way to store
/// metadata in library -- this trait just serves to decouple rustc_metadata from
/// the archive reader, which depends on LLVM.
pub trait MetadataLoader: std::fmt::Debug {
    fn get_rlib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String>;
    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String>;
}

pub type MetadataLoaderDyn = dyn MetadataLoader + Send + Sync;

/// A store of Rust crates, through which their metadata can be accessed.
///
/// Note that this trait should probably not be expanding today. All new
/// functionality should be driven through queries instead!
///
/// If you find a method on this trait named `{name}_untracked` it signifies
/// that it's *not* tracked for dependency information throughout compilation
/// (it'd break incremental compilation) and should only be called pre-HIR (e.g.
/// during resolve)
pub trait CrateStore: std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
    fn untracked_as_any(&mut self) -> &mut dyn Any;

    // Foreign definitions.
    // This information is safe to access, since it's hashed as part of the DefPathHash, which incr.
    // comp. uses to identify a DefId.
    fn def_key(&self, def: DefId) -> DefKey;
    fn def_path(&self, def: DefId) -> DefPath;
    fn def_path_hash(&self, def: DefId) -> DefPathHash;

    // This information is safe to access, since it's hashed as part of the StableCrateId, which
    // incr. comp. uses to identify a CrateNum.
    fn crate_name(&self, cnum: CrateNum) -> Symbol;
    fn stable_crate_id(&self, cnum: CrateNum) -> StableCrateId;
    fn stable_crate_id_to_crate_num(&self, stable_crate_id: StableCrateId) -> CrateNum;

    /// Fetch a DefId from a DefPathHash for a foreign crate.
    fn def_path_hash_to_def_id(&self, cnum: CrateNum, hash: DefPathHash) -> DefId;
    fn expn_hash_to_expn_id(
        &self,
        sess: &Session,
        cnum: CrateNum,
        index_guess: u32,
        hash: ExpnHash,
    ) -> ExpnId;

    /// Imports all `SourceFile`s from the given crate into the current session.
    /// This normally happens automatically when we decode a `Span` from
    /// that crate's metadata - however, the incr comp cache needs
    /// to trigger this manually when decoding a foreign `Span`
    fn import_source_files(&self, sess: &Session, cnum: CrateNum);
}

pub type CrateStoreDyn = dyn CrateStore + sync::Sync + sync::Send;

#[derive(Debug)]
pub struct Untracked {
    pub cstore: RwLock<Box<CrateStoreDyn>>,
    /// Reference span for definitions.
    pub source_span: RwLock<IndexVec<LocalDefId, Span>>,
    pub definitions: RwLock<Definitions>,
}
