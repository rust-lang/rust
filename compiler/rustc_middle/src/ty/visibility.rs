use rustc_hir::def_id::{DefId, LocalDefId};

use crate::ty::TyCtxt;

#[derive(Clone, Debug, PartialEq, Eq, Copy, Hash, Encodable, Decodable, HashStable)]
pub enum Visibility<Id = LocalDefId> {
    /// Visible everywhere (including in other crates).
    Public,
    /// Visible only in the given crate-local module.
    Restricted(Id),
}

impl<Id> Visibility<Id> {
    pub fn is_public(self) -> bool {
        matches!(self, Visibility::Public)
    }

    pub fn map_id<OutId>(self, f: impl FnOnce(Id) -> OutId) -> Visibility<OutId> {
        match self {
            Visibility::Public => Visibility::Public,
            Visibility::Restricted(id) => Visibility::Restricted(f(id)),
        }
    }
}

impl<Id: Into<DefId>> Visibility<Id> {
    pub fn to_def_id(self) -> Visibility<DefId> {
        self.map_id(Into::into)
    }

    /// Returns `true` if an item with this visibility is accessible from the given module.
    pub fn is_accessible_from(self, module: impl Into<DefId>, tcx: TyCtxt<'_>) -> bool {
        match self {
            // Public items are visible everywhere.
            Visibility::Public => true,
            Visibility::Restricted(id) => tcx.is_descendant_of(module.into(), id.into()),
        }
    }

    /// Returns `true` if this visibility is at least as accessible as the given visibility
    pub fn is_at_least(self, vis: Visibility<impl Into<DefId>>, tcx: TyCtxt<'_>) -> bool {
        match vis {
            Visibility::Public => self.is_public(),
            Visibility::Restricted(id) => self.is_accessible_from(id, tcx),
        }
    }
}

impl Visibility<DefId> {
    pub fn expect_local(self) -> Visibility {
        self.map_id(|id| id.expect_local())
    }

    /// Returns `true` if this item is visible anywhere in the local crate.
    pub fn is_visible_locally(self) -> bool {
        match self {
            Visibility::Public => true,
            Visibility::Restricted(def_id) => def_id.is_local(),
        }
    }
}
