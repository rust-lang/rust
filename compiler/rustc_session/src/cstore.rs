//! the rustc crate store interface. This also includes types that
//! are *mostly* used as a part of that interface, but these should
//! probably get a better home if someone can find one.

use std::any::Any;
use std::path::PathBuf;

use rustc_abi::ExternAbi;
use rustc_data_structures::sync::{self, AppendOnlyIndexVec, FreezeLock};
use rustc_hir::attrs::{CfgEntry, NativeLibKind, PeImportNameType};
use rustc_hir::def_id::{
    CrateNum, DefId, LOCAL_CRATE, LocalDefId, StableCrateId, StableCrateIdMap,
};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash, Definitions};
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::{Span, Symbol};

use crate::search_paths::PathKind;

// lonely orphan structs and enums looking for a better home

/// Where a crate came from on the local filesystem. One of these three options
/// must be non-None.
#[derive(PartialEq, Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct CrateSource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub rmeta: Option<(PathBuf, PathKind)>,
    pub sdylib_interface: Option<(PathBuf, PathKind)>,
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
    /// doesn't need to be linked to an rlib, e.g., the injected panic runtime.
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
    pub name: Symbol,
    /// If packed_bundled_libs enabled, actual filename of library is stored.
    pub filename: Option<Symbol>,
    pub cfg: Option<CfgEntry>,
    pub foreign_module: Option<DefId>,
    pub verbatim: Option<bool>,
    pub dll_imports: Vec<DllImport>,
}

impl NativeLib {
    pub fn has_modifiers(&self) -> bool {
        self.verbatim.is_some() || self.kind.has_modifiers()
    }

    pub fn wasm_import_module(&self) -> Option<Symbol> {
        if self.kind == NativeLibKind::WasmImportModule { Some(self.name) } else { None }
    }
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

    pub fn is_missing_decorations(&self) -> bool {
        self.import_name_type == Some(PeImportNameType::Undecorated)
            || self.import_name_type == Some(PeImportNameType::NoPrefix)
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
    pub abi: ExternAbi,
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
}

pub type CrateStoreDyn = dyn CrateStore + sync::DynSync + sync::DynSend;

pub struct Untracked {
    pub cstore: FreezeLock<Box<CrateStoreDyn>>,
    /// Reference span for definitions.
    pub source_span: AppendOnlyIndexVec<LocalDefId, Span>,
    pub definitions: FreezeLock<Definitions>,
    /// The interned [StableCrateId]s.
    pub stable_crate_ids: FreezeLock<StableCrateIdMap>,
}
