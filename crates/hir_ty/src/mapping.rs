//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_ir::{cast::Cast, fold::Shift, DebruijnIndex};
use chalk_solve::rust_ir;

use base_db::salsa::InternKey;
use hir_def::{GenericDefId, TypeAliasId};

use crate::{
    chalk_db, db::HirDatabase, AliasEq, AliasTy, CallableDefId, FnDefId, Interner, ProjectionTyExt,
    QuantifiedWhereClause, Substitution, Ty, WhereClause,
};

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

pub(super) fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substitution,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        result.push(pred.clone().substitute(&Interner, substs));
    }
    result
}

pub(super) fn generic_predicate_to_inline_bound(
    db: &dyn HirDatabase,
    pred: &QuantifiedWhereClause,
    self_ty: &Ty,
) -> Option<chalk_ir::Binders<rust_ir::InlineBound<Interner>>> {
    // An InlineBound is like a GenericPredicate, except the self type is left out.
    // We don't have a special type for this, but Chalk does.
    let self_ty_shifted_in = self_ty.clone().shifted_in_from(&Interner, DebruijnIndex::ONE);
    let (pred, binders) = pred.as_ref().into_value_and_skipped_binders();
    match pred {
        WhereClause::Implemented(trait_ref) => {
            if trait_ref.self_type_parameter(&Interner) != self_ty_shifted_in {
                // we can only convert predicates back to type bounds if they
                // have the expected self type
                return None;
            }
            let args_no_self = trait_ref.substitution.as_slice(&Interner)[1..]
                .iter()
                .map(|ty| ty.clone().cast(&Interner))
                .collect();
            let trait_bound = rust_ir::TraitBound { trait_id: trait_ref.trait_id, args_no_self };
            Some(chalk_ir::Binders::new(binders, rust_ir::InlineBound::TraitBound(trait_bound)))
        }
        WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
            if projection_ty.self_type_parameter(&Interner) != self_ty_shifted_in {
                return None;
            }
            let trait_ = projection_ty.trait_(db);
            let args_no_self = projection_ty.substitution.as_slice(&Interner)[1..]
                .iter()
                .map(|ty| ty.clone().cast(&Interner))
                .collect();
            let alias_eq_bound = rust_ir::AliasEqBound {
                value: ty.clone(),
                trait_bound: rust_ir::TraitBound { trait_id: trait_.to_chalk(db), args_no_self },
                associated_ty_id: projection_ty.associated_ty_id,
                parameters: Vec::new(), // FIXME we don't support generic associated types yet
            };
            Some(chalk_ir::Binders::new(
                binders,
                rust_ir::InlineBound::AliasEqBound(alias_eq_bound),
            ))
        }
        _ => None,
    }
}
