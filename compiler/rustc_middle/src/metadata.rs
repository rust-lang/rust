use rustc_hir::def::Res;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_span::Ident;
use rustc_span::def_id::DefId;
use smallvec::SmallVec;

use crate::ty;

/// A simplified version of `ImportKind` from resolve.
/// `DefId`s here correspond to `use` and `extern crate` items themselves, not their targets.
#[derive(Clone, Copy, Debug, TyEncodable, TyDecodable, HashStable)]
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

/// This structure is supposed to keep enough data to re-create `NameBinding`s for other crates
/// during name resolution. Right now the bindings are not recreated entirely precisely so we may
/// need to add more data in the future to correctly support macros 2.0, for example.
/// Module child can be either a proper item or a reexport (including private imports).
/// In case of reexport all the fields describe the reexport item itself, not what it refers to.
#[derive(Debug, TyEncodable, TyDecodable, HashStable)]
pub struct ModChild {
    /// Name of the item.
    pub ident: Ident,
    /// Resolution result corresponding to the item.
    /// Local variables cannot be exported, so this `Res` doesn't need the ID parameter.
    pub res: Res<!>,
    /// Visibility of the item.
    pub vis: ty::Visibility<DefId>,
    /// Reexport chain linking this module child to its original reexported item.
    /// Empty if the module child is a proper item.
    pub reexport_chain: SmallVec<[Reexport; 2]>,
}
