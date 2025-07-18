//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_solve::rust_ir;

use hir_def::{LifetimeParamId, TraitId, TypeAliasId, TypeOrConstParamId};
use salsa::{
    Id,
    plumbing::{AsId, FromId},
};

use crate::{
    AssocTypeId, CallableDefId, ChalkTraitId, FnDefId, ForeignDefId, Interner, OpaqueTyId,
    PlaceholderIndex, chalk_db,
    db::{HirDatabase, InternedLifetimeParamId, InternedTypeOrConstParamId},
};

pub trait ToChalk {
    type Chalk;
    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk;
    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self;
}

pub(crate) fn from_chalk<T, ChalkT>(db: &dyn HirDatabase, chalk: ChalkT) -> T
where
    T: ToChalk<Chalk = ChalkT>,
{
    T::from_chalk(db, chalk)
}

impl ToChalk for hir_def::ImplId {
    type Chalk = chalk_db::ImplId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::ImplId {
        chalk_ir::ImplId(self.as_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, impl_id: chalk_db::ImplId) -> hir_def::ImplId {
        FromId::from_id(impl_id.0.as_id())
    }
}

impl ToChalk for CallableDefId {
    type Chalk = FnDefId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> FnDefId {
        chalk_ir::FnDefId(salsa::plumbing::AsId::as_id(&self))
    }

    fn from_chalk(db: &dyn HirDatabase, fn_def_id: FnDefId) -> CallableDefId {
        salsa::plumbing::FromIdWithDb::from_id(fn_def_id.0, db.zalsa())
    }
}

pub(crate) struct TypeAliasAsValue(pub(crate) TypeAliasId);

impl ToChalk for TypeAliasAsValue {
    type Chalk = chalk_db::AssociatedTyValueId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::AssociatedTyValueId {
        rust_ir::AssociatedTyValueId(self.0.as_id())
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        assoc_ty_value_id: chalk_db::AssociatedTyValueId,
    ) -> TypeAliasAsValue {
        TypeAliasAsValue(TypeAliasId::from_id(assoc_ty_value_id.0))
    }
}

impl From<OpaqueTyId> for crate::db::InternedOpaqueTyId {
    fn from(id: OpaqueTyId) -> Self {
        FromId::from_id(id.0)
    }
}

impl From<crate::db::InternedOpaqueTyId> for OpaqueTyId {
    fn from(id: crate::db::InternedOpaqueTyId) -> Self {
        chalk_ir::OpaqueTyId(id.as_id())
    }
}

impl From<chalk_ir::ClosureId<Interner>> for crate::db::InternedClosureId {
    fn from(id: chalk_ir::ClosureId<Interner>) -> Self {
        FromId::from_id(id.0)
    }
}

impl From<crate::db::InternedClosureId> for chalk_ir::ClosureId<Interner> {
    fn from(id: crate::db::InternedClosureId) -> Self {
        chalk_ir::ClosureId(id.as_id())
    }
}

impl From<chalk_ir::CoroutineId<Interner>> for crate::db::InternedCoroutineId {
    fn from(id: chalk_ir::CoroutineId<Interner>) -> Self {
        Self::from_id(id.0)
    }
}

impl From<crate::db::InternedCoroutineId> for chalk_ir::CoroutineId<Interner> {
    fn from(id: crate::db::InternedCoroutineId) -> Self {
        chalk_ir::CoroutineId(id.as_id())
    }
}

pub fn to_foreign_def_id(id: TypeAliasId) -> ForeignDefId {
    chalk_ir::ForeignDefId(id.as_id())
}

pub fn from_foreign_def_id(id: ForeignDefId) -> TypeAliasId {
    FromId::from_id(id.0)
}

pub fn to_assoc_type_id(id: TypeAliasId) -> AssocTypeId {
    chalk_ir::AssocTypeId(id.as_id())
}

pub fn from_assoc_type_id(id: AssocTypeId) -> TypeAliasId {
    FromId::from_id(id.0)
}

pub fn from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> TypeOrConstParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    // SAFETY: We cannot really encapsulate this unfortunately, so just hope this is sound.
    let interned_id =
        InternedTypeOrConstParamId::from_id(unsafe { Id::from_index(idx.idx.try_into().unwrap()) });
    interned_id.loc(db)
}

pub fn to_placeholder_idx(db: &dyn HirDatabase, id: TypeOrConstParamId) -> PlaceholderIndex {
    let interned_id = InternedTypeOrConstParamId::new(db, id);
    PlaceholderIndex {
        ui: chalk_ir::UniverseIndex::ROOT,
        idx: interned_id.as_id().index() as usize,
    }
}

pub fn lt_from_placeholder_idx(db: &dyn HirDatabase, idx: PlaceholderIndex) -> LifetimeParamId {
    assert_eq!(idx.ui, chalk_ir::UniverseIndex::ROOT);
    // SAFETY: We cannot really encapsulate this unfortunately, so just hope this is sound.
    let interned_id =
        InternedLifetimeParamId::from_id(unsafe { Id::from_index(idx.idx.try_into().unwrap()) });
    interned_id.loc(db)
}

pub fn lt_to_placeholder_idx(db: &dyn HirDatabase, id: LifetimeParamId) -> PlaceholderIndex {
    let interned_id = InternedLifetimeParamId::new(db, id);
    PlaceholderIndex {
        ui: chalk_ir::UniverseIndex::ROOT,
        idx: interned_id.as_id().index() as usize,
    }
}

pub fn to_chalk_trait_id(id: TraitId) -> ChalkTraitId {
    chalk_ir::TraitId(id.as_id())
}

pub fn from_chalk_trait_id(id: ChalkTraitId) -> TraitId {
    FromId::from_id(id.0)
}
