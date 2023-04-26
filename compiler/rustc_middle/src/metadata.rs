use crate::ty;

use crate::ty::TyCtxt;
use rustc_hir::def::Res;
use rustc_macros::HashStable;
use rustc_span::def_id::DefId;
use rustc_span::symbol::{Ident, Symbol};
use smallvec::SmallVec;

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
pub struct ModChildData {
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

#[derive(Debug, TyEncodable, TyDecodable, HashStable)]
pub enum ModChild {
    Def(DefId),
    Reexport(ModChildData),
}

impl ModChild {
    pub fn res(&self, tcx: TyCtxt<'_>) -> Res<!> {
        match self {
            ModChild::Def(def_id) => Res::Def(tcx.def_kind(*def_id), *def_id),
            ModChild::Reexport(child) => child.res,
        }
    }

    pub fn opt_def_id(&self) -> Option<DefId> {
        match self {
            ModChild::Def(def_id) => Some(*def_id),
            ModChild::Reexport(child) => child.res.opt_def_id(),
        }
    }

    pub fn vis(&self, tcx: TyCtxt<'_>) -> ty::Visibility<DefId> {
        match self {
            ModChild::Def(def_id) => tcx.visibility(*def_id),
            ModChild::Reexport(child) => child.vis,
        }
    }

    pub fn name(&self, tcx: TyCtxt<'_>) -> Symbol {
        match self {
            ModChild::Def(def_id) => tcx.item_name(*def_id),
            ModChild::Reexport(child) => child.ident.name,
        }
    }

    // FIXME: `opt_item_ident` is buggy and doesn't decode hygiene correctly,
    // figure out what happens.
    pub fn ident(&self, tcx: TyCtxt<'_>) -> Ident {
        match self {
            ModChild::Def(def_id) => tcx.opt_item_ident(*def_id).unwrap(),
            ModChild::Reexport(child) => child.ident,
        }
    }

    pub fn reexport_chain(&self) -> &[Reexport] {
        match self {
            ModChild::Def(_) => &[],
            ModChild::Reexport(child) => &child.reexport_chain,
        }
    }
}
