//! the rustc crate store interface. This also includes types that
//! are *mostly* used as a part of that interface, but these should
//! probably get a better home if someone can find one.

use hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use hir::map as hir_map;
use hir::map::definitions::{DefKey, DefPathTable};
use rustc_data_structures::svh::Svh;
use ty::{self, TyCtxt};
use session::{Session, CrateDisambiguator};
use session::search_paths::PathKind;

use std::any::Any;
use std::path::{Path, PathBuf};
use syntax::ast;
use syntax::symbol::Symbol;
use syntax_pos::Span;
use rustc_target::spec::Target;
use rustc_data_structures::sync::{self, MetadataRef, Lrc};

pub use self::NativeLibraryKind::*;

// lonely orphan structs and enums looking for a better home

/// Where a crate came from on the local filesystem. One of these three options
/// must be non-None.
#[derive(PartialEq, Clone, Debug)]
pub struct CrateSource {
    pub dylib: Option<(PathBuf, PathKind)>,
    pub rlib: Option<(PathBuf, PathKind)>,
    pub rmeta: Option<(PathBuf, PathKind)>,
}

#[derive(RustcEncodable, RustcDecodable, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum DepKind {
    /// A dependency that is only used for its macros, none of which are visible from other crates.
    /// These are included in the metadata only as placeholders and are ignored when decoding.
    UnexportedMacrosOnly,
    /// A dependency that is only used for its macros.
    MacrosOnly,
    /// A dependency that is always injected into the dependency list and so
    /// doesn't need to be linked to an rlib, e.g., the injected allocator.
    Implicit,
    /// A dependency that is required by an rlib version of this crate.
    /// Ordinary `extern crate`s result in `Explicit` dependencies.
    Explicit,
}

impl DepKind {
    pub fn macros_only(self) -> bool {
        match self {
            DepKind::UnexportedMacrosOnly | DepKind::MacrosOnly => true,
            DepKind::Implicit | DepKind::Explicit => false,
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum LibSource {
    Some(PathBuf),
    MetadataOnly,
    None,
}

impl LibSource {
    pub fn is_some(&self) -> bool {
        if let LibSource::Some(_) = *self {
            true
        } else {
            false
        }
    }

    pub fn option(&self) -> Option<PathBuf> {
        match *self {
            LibSource::Some(ref p) => Some(p.clone()),
            LibSource::MetadataOnly | LibSource::None => None,
        }
    }
}

#[derive(Copy, Debug, PartialEq, Clone, RustcEncodable, RustcDecodable)]
pub enum LinkagePreference {
    RequireDynamic,
    RequireStatic,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub enum NativeLibraryKind {
    /// native static library (.a archive)
    NativeStatic,
    /// native static library, which doesn't get bundled into .rlibs
    NativeStaticNobundle,
    /// macOS-specific
    NativeFramework,
    /// default way to specify a dynamic library
    NativeUnknown,
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct NativeLibrary {
    pub kind: NativeLibraryKind,
    pub name: Option<Symbol>,
    pub cfg: Option<ast::MetaItem>,
    pub foreign_module: Option<DefId>,
    pub wasm_import_module: Option<Symbol>,
}

#[derive(Clone, Hash, RustcEncodable, RustcDecodable)]
pub struct ForeignModule {
    pub foreign_items: Vec<DefId>,
    pub def_id: DefId,
}

#[derive(Copy, Clone, Debug)]
pub struct ExternCrate {
    pub src: ExternCrateSource,

    /// span of the extern crate that caused this to be loaded
    pub span: Span,

    /// Number of links to reach the extern;
    /// used to select the extern with the shortest path
    pub path_len: usize,

    /// If true, then this crate is the crate named by the extern
    /// crate referenced above. If false, then this crate is a dep
    /// of the crate.
    pub direct: bool,
}

#[derive(Copy, Clone, Debug)]
pub enum ExternCrateSource {
    /// Crate is loaded by `extern crate`.
    Extern(
        /// def_id of the item in the current crate that caused
        /// this crate to be loaded; note that there could be multiple
        /// such ids
        DefId,
    ),
    // Crate is loaded by `use`.
    Use,
    /// Crate is implicitly loaded by an absolute or an `extern::` path.
    Path,
}

pub struct EncodedMetadata {
    pub raw_data: Vec<u8>
}

impl EncodedMetadata {
    pub fn new() -> EncodedMetadata {
        EncodedMetadata {
            raw_data: Vec::new(),
        }
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
    fn get_rlib_metadata(&self,
                         target: &Target,
                         filename: &Path)
                         -> Result<MetadataRef, String>;
    fn get_dylib_metadata(&self,
                          target: &Target,
                          filename: &Path)
                          -> Result<MetadataRef, String>;
}

/// A store of Rust crates, through with their metadata
/// can be accessed.
///
/// Note that this trait should probably not be expanding today. All new
/// functionality should be driven through queries instead!
///
/// If you find a method on this trait named `{name}_untracked` it signifies
/// that it's *not* tracked for dependency information throughout compilation
/// (it'd break incremental compilation) and should only be called pre-HIR (e.g.
/// during resolve)
pub trait CrateStore {
    fn crate_data_as_rc_any(&self, krate: CrateNum) -> Lrc<dyn Any>;

    // resolve
    fn def_key(&self, def: DefId) -> DefKey;
    fn def_path(&self, def: DefId) -> hir_map::DefPath;
    fn def_path_hash(&self, def: DefId) -> hir_map::DefPathHash;
    fn def_path_table(&self, cnum: CrateNum) -> Lrc<DefPathTable>;

    // "queries" used in resolve that aren't tracked for incremental compilation
    fn crate_name_untracked(&self, cnum: CrateNum) -> Symbol;
    fn crate_disambiguator_untracked(&self, cnum: CrateNum) -> CrateDisambiguator;
    fn crate_hash_untracked(&self, cnum: CrateNum) -> Svh;
    fn extern_mod_stmt_cnum_untracked(&self, emod_id: ast::NodeId) -> Option<CrateNum>;
    fn item_generics_cloned_untracked(&self, def: DefId, sess: &Session) -> ty::Generics;
    fn postorder_cnums_untracked(&self) -> Vec<CrateNum>;

    // This is basically a 1-based range of ints, which is a little
    // silly - I may fix that.
    fn crates_untracked(&self) -> Vec<CrateNum>;

    // utility functions
    fn encode_metadata<'a, 'tcx>(&self,
                                 tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                 -> EncodedMetadata;
    fn metadata_encoding_version(&self) -> &[u8];
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
pub fn used_crates(tcx: TyCtxt<'_, '_, '_>, prefer: LinkagePreference)
    -> Vec<(CrateNum, LibSource)>
{
    let mut libs = tcx.crates()
        .iter()
        .cloned()
        .filter_map(|cnum| {
            if tcx.dep_kind(cnum).macros_only() {
                return None
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
    let mut ordering = tcx.postorder_cnums(LOCAL_CRATE);
    Lrc::make_mut(&mut ordering).reverse();
    libs.sort_by_cached_key(|&(a, _)| {
        ordering.iter().position(|x| *x == a)
    });
    libs
}
