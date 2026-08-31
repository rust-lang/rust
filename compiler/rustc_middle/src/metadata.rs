use rustc_hir::def::Res;
use rustc_macros::{StableHash, TyDecodable, TyEncodable};
use rustc_span::Ident;
use rustc_span::def_id::{DefId, ModId};
use rustc_span::edition::Edition;
use smallvec::SmallVec;

use crate::ty;

/// A simplified version of `ImportKind` from resolve.
/// `DefId`s here correspond to `use` and `extern crate` items themselves, not their targets.
#[derive(Clone, Copy, Debug, TyEncodable, TyDecodable, StableHash)]
pub enum Reexport {
    Single(DefId),
    Glob(DefId),
    ExternCrate(DefId),
    MacroUse,
    MacroExport,
}

impl Reexport {
    pub fn id(self) -> Option<DefId> {
        match self {
            Reexport::Single(id) | Reexport::Glob(id) | Reexport::ExternCrate(id) => Some(id),
            Reexport::MacroUse | Reexport::MacroExport => None,
        }
    }
}

/// A different item that a module child resolves to on or before a given edition.
#[derive(Clone, Copy, Debug, TyEncodable, TyDecodable, StableHash)]
pub struct EditionRedirect {
    pub edition: Edition,
    pub target: Res<!>,
}

/// This structure is supposed to keep enough data to re-create `Decl`s for other crates
/// during name resolution. Right now the bindings are not recreated entirely precisely so we may
/// need to add more data in the future to correctly support macros 2.0, for example.
/// Module child can be either a proper item or a reexport (including private imports).
/// In case of reexport all the fields describe the reexport item itself, not what it refers to.
#[derive(Debug, TyEncodable, TyDecodable, StableHash)]
pub struct ModChild {
    /// Name of the item.
    pub ident: Ident,
    /// Resolution result corresponding to the item.
    /// Local variables cannot be exported, so this `Res` doesn't need the ID parameter.
    pub res: Res<!>,
    /// Visibility of the item.
    pub vis: ty::Visibility<ModId>,
    /// Reexport chain linking this module child to its original reexported item.
    /// Empty if the module child is a proper item.
    pub reexport_chain: SmallVec<[Reexport; 2]>,
    /// Edition-dependent alternatives, sorted from the earliest boundary to the latest.
    pub edition_redirects: SmallVec<[EditionRedirect; 1]>,
}

/// Same as `ModChild`, however, it includes ambiguity error.
#[derive(Debug, TyEncodable, TyDecodable, StableHash)]
pub struct AmbigModChild {
    pub main: ModChild,
    pub second: ModChild,
}
