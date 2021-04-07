//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_ir::{cast::Cast, interner::HasInterner};
use chalk_solve::rust_ir;

use base_db::salsa::InternKey;
use hir_def::{GenericDefId, TypeAliasId};

use crate::{
    db::HirDatabase, static_lifetime, AliasTy, CallableDefId, Canonical, ConstrainedSubst,
    DomainGoal, FnPointer, GenericArg, InEnvironment, OpaqueTy, ProjectionTy, ProjectionTyExt,
    QuantifiedWhereClause, Substitution, TraitRef, Ty, TypeWalk, WhereClause,
};

use super::interner::*;
use super::*;

impl ToChalk for Ty {
    type Chalk = chalk_ir::Ty<Interner>;
    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Ty<Interner> {
        self
    }
    fn from_chalk(db: &dyn HirDatabase, chalk: chalk_ir::Ty<Interner>) -> Self {
        chalk
    }
}

impl ToChalk for GenericArg {
    type Chalk = chalk_ir::GenericArg<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        chalk
    }
}

impl ToChalk for Substitution {
    type Chalk = chalk_ir::Substitution<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Substitution<Interner> {
        self
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        parameters: chalk_ir::Substitution<Interner>,
    ) -> Substitution {
        parameters
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef<Interner>;

    fn to_chalk(self: TraitRef, db: &dyn HirDatabase) -> chalk_ir::TraitRef<Interner> {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, trait_ref: chalk_ir::TraitRef<Interner>) -> Self {
        trait_ref
    }
}

impl ToChalk for hir_def::TraitId {
    type Chalk = TraitId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> TraitId {
        chalk_ir::TraitId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, trait_id: TraitId) -> hir_def::TraitId {
        InternKey::from_intern_id(trait_id.0)
    }
}

impl ToChalk for hir_def::ImplId {
    type Chalk = ImplId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> ImplId {
        chalk_ir::ImplId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, impl_id: ImplId) -> hir_def::ImplId {
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
    type Chalk = AssociatedTyValueId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> AssociatedTyValueId {
        rust_ir::AssociatedTyValueId(self.0.as_intern_id())
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        assoc_ty_value_id: AssociatedTyValueId,
    ) -> TypeAliasAsValue {
        TypeAliasAsValue(TypeAliasId::from_intern_id(assoc_ty_value_id.0))
    }
}

impl ToChalk for WhereClause {
    type Chalk = chalk_ir::WhereClause<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::WhereClause<Interner> {
        self
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        where_clause: chalk_ir::WhereClause<Interner>,
    ) -> WhereClause {
        where_clause
    }
}

impl ToChalk for ProjectionTy {
    type Chalk = chalk_ir::ProjectionTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::ProjectionTy<Interner> {
        self
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<Interner>,
    ) -> ProjectionTy {
        projection_ty
    }
}
impl ToChalk for OpaqueTy {
    type Chalk = chalk_ir::OpaqueTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        chalk
    }
}

impl ToChalk for AliasTy {
    type Chalk = chalk_ir::AliasTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        chalk
    }
}

impl ToChalk for AliasEq {
    type Chalk = chalk_ir::AliasEq<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::AliasEq<Interner> {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, alias_eq: chalk_ir::AliasEq<Interner>) -> Self {
        alias_eq
    }
}

impl ToChalk for DomainGoal {
    type Chalk = chalk_ir::DomainGoal<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::DomainGoal<Interner> {
        self
    }

    fn from_chalk(_db: &dyn HirDatabase, goal: chalk_ir::DomainGoal<Interner>) -> Self {
        goal
    }
}

impl<T> ToChalk for Canonical<T>
where
    T: HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::Canonical<T>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Canonical<T> {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, canonical: chalk_ir::Canonical<T>) -> Canonical<T> {
        canonical
    }
}

impl<T: ToChalk> ToChalk for InEnvironment<T>
where
    T: HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::InEnvironment<T>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::InEnvironment<T> {
        self
    }

    fn from_chalk(_db: &dyn HirDatabase, in_env: chalk_ir::InEnvironment<T>) -> InEnvironment<T> {
        in_env
    }
}

impl<T: ToChalk> ToChalk for crate::Binders<T>
where
    T: HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::Binders<T>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Binders<T> {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, binders: chalk_ir::Binders<T>) -> crate::Binders<T> {
        binders
    }
}

impl ToChalk for crate::ConstrainedSubst {
    type Chalk = chalk_ir::ConstrainedSubst<Interner>;

    fn to_chalk(self, _db: &dyn HirDatabase) -> Self::Chalk {
        self
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        chalk
    }
}

pub(super) fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T>
where
    T: HasInterner<Interner = Interner>,
{
    chalk_ir::Binders::new(
        chalk_ir::VariableKinds::from_iter(
            &Interner,
            std::iter::repeat(chalk_ir::VariableKind::Ty(chalk_ir::TyVariableKind::General))
                .take(num_vars),
        ),
        value,
    )
}

pub(super) fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substitution,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        result.push(pred.clone().substitute(&Interner, substs).to_chalk(db));
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
            let args_no_self = trait_ref.substitution.interned()[1..]
                .iter()
                .map(|ty| ty.clone().to_chalk(db).cast(&Interner))
                .collect();
            let trait_bound = rust_ir::TraitBound { trait_id: trait_ref.trait_id, args_no_self };
            Some(chalk_ir::Binders::new(binders, rust_ir::InlineBound::TraitBound(trait_bound)))
        }
        WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
            if projection_ty.self_type_parameter(&Interner) != self_ty_shifted_in {
                return None;
            }
            let trait_ = projection_ty.trait_(db);
            let args_no_self = projection_ty.substitution.interned()[1..]
                .iter()
                .map(|ty| ty.clone().to_chalk(db).cast(&Interner))
                .collect();
            let alias_eq_bound = rust_ir::AliasEqBound {
                value: ty.clone().to_chalk(db),
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
