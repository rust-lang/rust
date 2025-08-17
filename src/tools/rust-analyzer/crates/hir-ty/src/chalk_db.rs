//! The implementation of `RustIrDatabase` for Chalk, which provides information
//! about the code that Chalk needs.
use hir_def::{CallableDefId, GenericDefId};

use crate::{Interner, Substitution, db::HirDatabase, mapping::from_chalk};

pub(crate) type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub(crate) type TraitId = chalk_ir::TraitId<Interner>;
pub(crate) type AdtId = chalk_ir::AdtId<Interner>;
pub(crate) type ImplId = chalk_ir::ImplId<Interner>;
pub(crate) type Variances = chalk_ir::Variances<Interner>;

impl chalk_ir::UnificationDatabase<Interner> for &dyn HirDatabase {
    fn fn_def_variance(
        &self,
        fn_def_id: chalk_ir::FnDefId<Interner>,
    ) -> chalk_ir::Variances<Interner> {
        HirDatabase::fn_def_variance(*self, from_chalk(*self, fn_def_id))
    }

    fn adt_variance(&self, adt_id: chalk_ir::AdtId<Interner>) -> chalk_ir::Variances<Interner> {
        HirDatabase::adt_variance(*self, adt_id.0)
    }
}

pub(crate) fn fn_def_variance_query(
    db: &dyn HirDatabase,
    callable_def: CallableDefId,
) -> Variances {
    Variances::from_iter(
        Interner,
        db.variances_of(GenericDefId::from_callable(db, callable_def))
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|v| match v {
                crate::variance::Variance::Covariant => chalk_ir::Variance::Covariant,
                crate::variance::Variance::Invariant => chalk_ir::Variance::Invariant,
                crate::variance::Variance::Contravariant => chalk_ir::Variance::Contravariant,
                crate::variance::Variance::Bivariant => chalk_ir::Variance::Invariant,
            }),
    )
}

pub(crate) fn adt_variance_query(db: &dyn HirDatabase, adt_id: hir_def::AdtId) -> Variances {
    Variances::from_iter(
        Interner,
        db.variances_of(adt_id.into()).as_deref().unwrap_or_default().iter().map(|v| match v {
            crate::variance::Variance::Covariant => chalk_ir::Variance::Covariant,
            crate::variance::Variance::Invariant => chalk_ir::Variance::Invariant,
            crate::variance::Variance::Contravariant => chalk_ir::Variance::Contravariant,
            crate::variance::Variance::Bivariant => chalk_ir::Variance::Invariant,
        }),
    )
}

/// Returns instantiated predicates.
pub(super) fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substitution,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    db.generic_predicates(def)
        .iter()
        .cloned()
        .map(|pred| pred.substitute(Interner, substs))
        .collect()
}
