//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_solve::rust_ir;

use base_db::salsa::InternKey;
use hir_def::TypeAliasId;

use crate::{chalk_db, db::HirDatabase, CallableDefId, FnDefId, Interner, OpaqueTyId};

pub(crate) trait ToChalk {
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

impl ToChalk for hir_def::TraitId {
    type Chalk = chalk_db::TraitId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::TraitId {
        chalk_ir::TraitId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, trait_id: chalk_db::TraitId) -> hir_def::TraitId {
        InternKey::from_intern_id(trait_id.0)
    }
}

impl ToChalk for hir_def::ImplId {
    type Chalk = chalk_db::ImplId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::ImplId {
        chalk_ir::ImplId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, impl_id: chalk_db::ImplId) -> hir_def::ImplId {
        InternKey::from_intern_id(impl_id.0)
    }
}

impl ToChalk for CallableDefId {
    type Chalk = FnDefId;

    fn to_chalk(self, db: &dyn HirDatabase) -> FnDefId {
        db.intern_callable_def(self).into()
    }

    fn from_chalk(db: &dyn HirDatabase, fn_def_id: FnDefId) -> CallableDefId {
        db.lookup_intern_callable_def(fn_def_id.into())
    }
}

pub(crate) struct TypeAliasAsValue(pub(crate) TypeAliasId);

impl ToChalk for TypeAliasAsValue {
    type Chalk = chalk_db::AssociatedTyValueId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> chalk_db::AssociatedTyValueId {
        rust_ir::AssociatedTyValueId(self.0.as_intern_id())
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        assoc_ty_value_id: chalk_db::AssociatedTyValueId,
    ) -> TypeAliasAsValue {
        TypeAliasAsValue(TypeAliasId::from_intern_id(assoc_ty_value_id.0))
    }
}

impl From<FnDefId> for crate::db::InternedCallableDefId {
    fn from(fn_def_id: FnDefId) -> Self {
        InternKey::from_intern_id(fn_def_id.0)
    }
}

impl From<crate::db::InternedCallableDefId> for FnDefId {
    fn from(callable_def_id: crate::db::InternedCallableDefId) -> Self {
        chalk_ir::FnDefId(callable_def_id.as_intern_id())
    }
}

impl From<OpaqueTyId> for crate::db::InternedOpaqueTyId {
    fn from(id: OpaqueTyId) -> Self {
        InternKey::from_intern_id(id.0)
    }
}

impl From<crate::db::InternedOpaqueTyId> for OpaqueTyId {
    fn from(id: crate::db::InternedOpaqueTyId) -> Self {
        chalk_ir::OpaqueTyId(id.as_intern_id())
    }
}

impl From<chalk_ir::ClosureId<Interner>> for crate::db::InternedClosureId {
    fn from(id: chalk_ir::ClosureId<Interner>) -> Self {
        Self::from_intern_id(id.0)
    }
}

impl From<crate::db::InternedClosureId> for chalk_ir::ClosureId<Interner> {
    fn from(id: crate::db::InternedClosureId) -> Self {
        chalk_ir::ClosureId(id.as_intern_id())
    }
}
