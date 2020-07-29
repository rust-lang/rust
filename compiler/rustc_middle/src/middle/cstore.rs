//! the rustc crate store interface. This also includes types that
//! are *mostly* used as a part of that interface, but these should
//! probably get a better home if someone can find one.

use crate::ty::TyCtxt;

use rustc_ast as ast;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{self, MetadataRef};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathHash};
use rustc_macros::HashStable;
use rustc_session::search_paths::PathKind;
use rustc_session::utils::NativeLibKind;
use rustc_session::CrateDisambiguator;
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use rustc_target::spec::Target;

use std::any::Any;
use std::path::{Path, PathBuf};

// lonely orphan structs and enums looking for a better home

/// Where a crate came from on the local filesystem. One of these three options
/// must be non-None.
#[derive(PartialEq, Clone, Debug, HashStable, Encodable, Decodable)]
pub struct CrateSource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub rmeta: Option<(PathBuf, PathKind)>,
}

impl CrateSource {
    pub fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.dylib.iter().chain(self.rlib.iter()).chain(self.rmeta.iter()).map(|p| &p.0)
    }
}

#[derive(Encodable, Decodable, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
#[derive(HashStable)]
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
    pub fn macros_only(self) -> bool {
        match self {
            CrateDepKind::MacrosOnly => true,
            CrateDepKind::Implicit | CrateDepKind::Explicit => false,
        }
    }
}

#[derive(PartialEq, Clone, Debug, Encodable, Decodable)]
pub enum LibSource {
    Some(PathBuf),
    MetadataOnly,
    None,
}

impl LibSource {
    pub fn is_some(&self) -> bool {
        matches!(self, LibSource::Some(_))
    }

    pub fn option(&self) -> Option<PathBuf> {
        match *self {
            LibSource::Some(ref p) => Some(p.clone()),
            LibSource::MetadataOnly | LibSource::None => None,
        }
    }
}

#[derive(Copy, Debug, PartialEq, Clone, Encodable, Decodable, HashStable)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

#[derive(Clone, Debug, Encodable, Decodable, HashStable)]
pub struct NativeLib {
    pub kind: NativeLibKind,
    pub name: Option<Symbol>,
    pub cfg: Option<ast::MetaItem>,
    pub foreign_module: Option<DefId>,
    pub wasm_import_module: Option<Symbol>,
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable)]
pub struct ForeignModule {
    pub foreign_items: Vec<DefId>,
    pub def_id: DefId,
}

#[derive(Copy, Clone, Debug, HashStable)]
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
    pub fn is_direct(&self) -> bool {
        self.dependency_of == LOCAL_CRATE
    }

    pub fn rank(&self) -> impl PartialOrd {
        // Prefer:
        // - direct extern crate to indirect
        // - shorter paths to longer
        (self.is_direct(), !self.path_len)
    }
}

#[derive(Copy, Clone, Debug, HashStable)]
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

#[derive(Encodable, Decodable)]
pub struct EncodedMetadata {
    pub raw_data: Vec<u8>,
}

impl EncodedMetadata {
    pub fn new() -> EncodedMetadata {
        EncodedMetadata { raw_data: Vec::new() }
    }
}

/// The backend's way to give the crate store access to the metadata in a library.
/// Note that it returns the raw metadata bytes stored in the library file, whether
/// it is compressed, uncompressed, some weird mix, etc.
/// rmeta files are backend independent and not handled here.
///
/// At the time of this writing, there is only one backend and one way to store
/// metadata in library -- this trait just serves to decouple rustc_metadata from
/// the archive reader, which depends on LLVM.
pub trait MetadataLoader {
    fn get_rlib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String>;
    fn get_dylib_metadata(&self, target: &Target, filename: &Path) -> Result<MetadataRef, String>;
}

pub type MetadataLoaderDyn = dyn MetadataLoader + Sync;

/// A store of Rust crates, through which their metadata can be accessed.
///
/// Note that this trait should probably not be expanding today. All new
/// functionality should be driven through queries instead!
///
/// If you find a method on this trait named `{name}_untracked` it signifies
/// that it's *not* tracked for dependency information throughout compilation
/// (it'd break incremental compilation) and should only be called pre-HIR (e.g.
/// during resolve)
pub trait CrateStore {
    fn as_any(&self) -> &dyn Any;

    // resolve
    fn def_key(&self, def: DefId) -> DefKey;
    fn def_kind(&self, def: DefId) -> DefKind;
    fn def_path(&self, def: DefId) -> DefPath;
    fn def_path_hash(&self, def: DefId) -> DefPathHash;
    fn all_def_path_hashes_and_def_ids(&self, cnum: CrateNum) -> Vec<(DefPathHash, DefId)>;
    fn num_def_ids(&self, cnum: CrateNum) -> usize;
    fn def_path_hash_to_def_id(
        &self,
        cnum: CrateNum,
        index_guess: u32,
        hash: DefPathHash,
    ) -> Option<DefId>;

    // "queries" used in resolve that aren't tracked for incremental compilation
    fn crate_name_untracked(&self, cnum: CrateNum) -> Symbol;
    fn crate_is_private_dep_untracked(&self, cnum: CrateNum) -> bool;
    fn crate_disambiguator_untracked(&self, cnum: CrateNum) -> CrateDisambiguator;
    fn crate_hash_untracked(&self, cnum: CrateNum) -> Svh;

    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates_untracked(&self) -> Vec<CrateNum>;

    // utility functions
    fn encode_metadata(&self, tcx: TyCtxt<'_>) -> EncodedMetadata;
    fn metadata_encoding_version(&self) -> &[u8];
    fn allocator_kind(&self) -> Option<AllocatorKind>;
}

pub type CrateStoreDyn = dyn CrateStore + sync::Sync;

// This method is used when generating the command line to pass through to
// system linker. The linker expects undefined symbols on the left of the
// command line to be defined in libraries on the right, not the other way
// around. For more info, see some comments in the add_used_library function
// below.
//
// In order to get this left-to-right dependency ordering, we perform a
// topological sort of all crates putting the leaves at the right-most
// positions.
pub fn used_crates(tcx: TyCtxt<'_>, prefer: LinkagePreference) -> Vec<(CrateNum, LibSource)> {
    let mut libs = tcx
        .crates()
        .iter()
        .cloned()
        .filter_map(|cnum| {
            if tcx.dep_kind(cnum).macros_only() {
                return None;
            }
            let source = tcx.used_crate_source(cnum);
            let path = match prefer {
                LinkagePreference::RequireDynamic => source.dylib.clone().map(|p| p.0),
                LinkagePreference::RequireStatic => source.rlib.clone().map(|p| p.0),
            };
            let path = match path {
                Some(p) => LibSource::Some(p),
                None => {
                    if source.rmeta.is_some() {
                        LibSource::MetadataOnly
                    } else {
                        LibSource::None
                    }
                }
            };
            Some((cnum, path))
        })
        .collect::<Vec<_>>();
    let mut ordering = tcx.postorder_cnums(LOCAL_CRATE).to_owned();
    ordering.reverse();
    libs.sort_by_cached_key(|&(a, _)| ordering.iter().position(|x| *x == a));
    libs
}
