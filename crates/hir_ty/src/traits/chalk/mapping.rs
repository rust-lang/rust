//! This module contains the implementations of the `ToChalk` trait, which
//! handles conversion between our data types and their corresponding types in
//! Chalk (in both directions); plus some helper functions for more specialized
//! conversions.

use chalk_ir::{cast::Cast, fold::shift::Shift, interner::HasInterner, LifetimeData};
use chalk_solve::rust_ir;

use base_db::salsa::InternKey;
use hir_def::{GenericDefId, TypeAliasId};

use crate::{
    db::HirDatabase, primitive::UintTy, AliasTy, CallableDefId, Canonical, DomainGoal, FnPointer,
    GenericArg, InEnvironment, OpaqueTy, ProjectionTy, QuantifiedWhereClause, Scalar, Substitution,
    TraitRef, Ty, TypeWalk, WhereClause,
};

use super::interner::*;
use super::*;

impl ToChalk for Ty {
    type Chalk = chalk_ir::Ty<Interner>;
    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Ty<Interner> {
        match self.into_inner() {
            TyKind::Ref(m, ty) => ref_to_chalk(db, m, ty),
            TyKind::Array(ty) => array_to_chalk(db, ty),
            TyKind::Function(FnPointer { sig, substs, .. }) => {
                let substitution = chalk_ir::FnSubst(substs.to_chalk(db).shifted_in(&Interner));
                chalk_ir::TyKind::Function(chalk_ir::FnPointer {
                    num_binders: 0,
                    sig,
                    substitution,
                })
                .intern(&Interner)
            }
            TyKind::AssociatedType(assoc_type_id, substs) => {
                let substitution = substs.to_chalk(db);
                chalk_ir::TyKind::AssociatedType(assoc_type_id, substitution).intern(&Interner)
            }

            TyKind::OpaqueType(id, substs) => {
                let substitution = substs.to_chalk(db);
                chalk_ir::TyKind::OpaqueType(id, substitution).intern(&Interner)
            }

            TyKind::Foreign(id) => chalk_ir::TyKind::Foreign(id).intern(&Interner),

            TyKind::Scalar(scalar) => chalk_ir::TyKind::Scalar(scalar).intern(&Interner),

            TyKind::Tuple(cardinality, substs) => {
                let substitution = substs.to_chalk(db);
                chalk_ir::TyKind::Tuple(cardinality, substitution).intern(&Interner)
            }
            TyKind::Raw(mutability, ty) => {
                let ty = ty.to_chalk(db);
                chalk_ir::TyKind::Raw(mutability, ty).intern(&Interner)
            }
            TyKind::Slice(ty) => chalk_ir::TyKind::Slice(ty.to_chalk(db)).intern(&Interner),
            TyKind::Str => chalk_ir::TyKind::Str.intern(&Interner),
            TyKind::FnDef(id, substs) => {
                let substitution = substs.to_chalk(db);
                chalk_ir::TyKind::FnDef(id, substitution).intern(&Interner)
            }
            TyKind::Never => chalk_ir::TyKind::Never.intern(&Interner),

            TyKind::Closure(closure_id, substs) => {
                let substitution = substs.to_chalk(db);
                chalk_ir::TyKind::Closure(closure_id, substitution).intern(&Interner)
            }

            TyKind::Adt(adt_id, substs) => {
                let substitution = substs.to_chalk(db);
                chalk_ir::TyKind::Adt(adt_id, substitution).intern(&Interner)
            }
            TyKind::Alias(AliasTy::Projection(proj_ty)) => {
                let associated_ty_id = proj_ty.associated_ty_id;
                let substitution = proj_ty.substitution.to_chalk(db);
                chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                    associated_ty_id,
                    substitution,
                })
                .cast(&Interner)
                .intern(&Interner)
            }
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                let opaque_ty_id = opaque_ty.opaque_ty_id;
                let substitution = opaque_ty.substitution.to_chalk(db);
                chalk_ir::AliasTy::Opaque(chalk_ir::OpaqueTy { opaque_ty_id, substitution })
                    .cast(&Interner)
                    .intern(&Interner)
            }
            TyKind::Placeholder(idx) => idx.to_ty::<Interner>(&Interner),
            TyKind::BoundVar(idx) => chalk_ir::TyKind::BoundVar(idx).intern(&Interner),
            TyKind::InferenceVar(..) => panic!("uncanonicalized infer ty"),
            TyKind::Dyn(dyn_ty) => {
                let (bounds, binders) = dyn_ty.bounds.into_value_and_skipped_binders();
                let where_clauses = chalk_ir::QuantifiedWhereClauses::from_iter(
                    &Interner,
                    bounds.interned().iter().cloned().map(|p| p.to_chalk(db)),
                );
                let bounded_ty = chalk_ir::DynTy {
                    bounds: chalk_ir::Binders::new(binders, where_clauses),
                    lifetime: LifetimeData::Static.intern(&Interner),
                };
                chalk_ir::TyKind::Dyn(bounded_ty).intern(&Interner)
            }
            TyKind::Error => chalk_ir::TyKind::Error.intern(&Interner),
        }
    }
    fn from_chalk(db: &dyn HirDatabase, chalk: chalk_ir::Ty<Interner>) -> Self {
        match chalk.data(&Interner).kind.clone() {
            chalk_ir::TyKind::Error => TyKind::Error,
            chalk_ir::TyKind::Array(ty, _size) => TyKind::Array(from_chalk(db, ty)),
            chalk_ir::TyKind::Placeholder(idx) => TyKind::Placeholder(idx),
            chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Projection(proj)) => {
                let associated_ty = proj.associated_ty_id;
                let parameters = from_chalk(db, proj.substitution);
                TyKind::Alias(AliasTy::Projection(ProjectionTy {
                    associated_ty_id: associated_ty,
                    substitution: parameters,
                }))
            }
            chalk_ir::TyKind::Alias(chalk_ir::AliasTy::Opaque(opaque_ty)) => {
                let opaque_ty_id = opaque_ty.opaque_ty_id;
                let parameters = from_chalk(db, opaque_ty.substitution);
                TyKind::Alias(AliasTy::Opaque(OpaqueTy { opaque_ty_id, substitution: parameters }))
            }
            chalk_ir::TyKind::Function(chalk_ir::FnPointer {
                num_binders,
                sig,
                substitution,
                ..
            }) => {
                assert_eq!(num_binders, 0);
                let substs: Substitution = from_chalk(
                    db,
                    substitution.0.shifted_out(&Interner).expect("fn ptr should have no binders"),
                );
                TyKind::Function(FnPointer { num_args: (substs.len(&Interner) - 1), sig, substs })
            }
            chalk_ir::TyKind::BoundVar(idx) => TyKind::BoundVar(idx),
            chalk_ir::TyKind::InferenceVar(_iv, _kind) => TyKind::Error,
            chalk_ir::TyKind::Dyn(where_clauses) => {
                assert_eq!(where_clauses.bounds.binders.len(&Interner), 1);
                let bounds = where_clauses
                    .bounds
                    .skip_binders()
                    .iter(&Interner)
                    .map(|c| from_chalk(db, c.clone()));
                TyKind::Dyn(crate::DynTy {
                    bounds: crate::Binders::new(
                        where_clauses.bounds.binders.clone(),
                        crate::QuantifiedWhereClauses::from_iter(&Interner, bounds),
                    ),
                })
            }

            chalk_ir::TyKind::Adt(adt_id, subst) => TyKind::Adt(adt_id, from_chalk(db, subst)),
            chalk_ir::TyKind::AssociatedType(type_id, subst) => {
                TyKind::AssociatedType(type_id, from_chalk(db, subst))
            }

            chalk_ir::TyKind::OpaqueType(opaque_type_id, subst) => {
                TyKind::OpaqueType(opaque_type_id, from_chalk(db, subst))
            }

            chalk_ir::TyKind::Scalar(scalar) => TyKind::Scalar(scalar),
            chalk_ir::TyKind::Tuple(cardinality, subst) => {
                TyKind::Tuple(cardinality, from_chalk(db, subst))
            }
            chalk_ir::TyKind::Raw(mutability, ty) => TyKind::Raw(mutability, from_chalk(db, ty)),
            chalk_ir::TyKind::Slice(ty) => TyKind::Slice(from_chalk(db, ty)),
            chalk_ir::TyKind::Ref(mutability, _lifetime, ty) => {
                TyKind::Ref(mutability, from_chalk(db, ty))
            }
            chalk_ir::TyKind::Str => TyKind::Str,
            chalk_ir::TyKind::Never => TyKind::Never,

            chalk_ir::TyKind::FnDef(fn_def_id, subst) => {
                TyKind::FnDef(fn_def_id, from_chalk(db, subst))
            }

            chalk_ir::TyKind::Closure(id, subst) => TyKind::Closure(id, from_chalk(db, subst)),

            chalk_ir::TyKind::Foreign(foreign_def_id) => TyKind::Foreign(foreign_def_id),
            chalk_ir::TyKind::Generator(_, _) => unimplemented!(), // FIXME
            chalk_ir::TyKind::GeneratorWitness(_, _) => unimplemented!(), // FIXME
        }
        .intern(&Interner)
    }
}

/// We currently don't model lifetimes, but Chalk does. So, we have to insert a
/// fake lifetime here, because Chalks built-in logic may expect it to be there.
fn ref_to_chalk(
    db: &dyn HirDatabase,
    mutability: chalk_ir::Mutability,
    ty: Ty,
) -> chalk_ir::Ty<Interner> {
    let arg = ty.to_chalk(db);
    let lifetime = LifetimeData::Static.intern(&Interner);
    chalk_ir::TyKind::Ref(mutability, lifetime, arg).intern(&Interner)
}

/// We currently don't model constants, but Chalk does. So, we have to insert a
/// fake constant here, because Chalks built-in logic may expect it to be there.
fn array_to_chalk(db: &dyn HirDatabase, ty: Ty) -> chalk_ir::Ty<Interner> {
    let arg = ty.to_chalk(db);
    let usize_ty = chalk_ir::TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(&Interner);
    let const_ = chalk_ir::ConstData {
        ty: usize_ty,
        value: chalk_ir::ConstValue::Concrete(chalk_ir::ConcreteConst { interned: () }),
    }
    .intern(&Interner);
    chalk_ir::TyKind::Array(arg, const_).intern(&Interner)
}

impl ToChalk for GenericArg {
    type Chalk = chalk_ir::GenericArg<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk {
        match self.interned() {
            crate::GenericArgData::Ty(ty) => ty.clone().to_chalk(db).cast(&Interner),
        }
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        match chalk.interned() {
            chalk_ir::GenericArgData::Ty(ty) => Ty::from_chalk(db, ty.clone()).cast(&Interner),
            chalk_ir::GenericArgData::Lifetime(_) => unimplemented!(),
            chalk_ir::GenericArgData::Const(_) => unimplemented!(),
        }
    }
}

impl ToChalk for Substitution {
    type Chalk = chalk_ir::Substitution<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Substitution<Interner> {
        chalk_ir::Substitution::from_iter(
            &Interner,
            self.iter(&Interner).map(|ty| ty.clone().to_chalk(db)),
        )
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        parameters: chalk_ir::Substitution<Interner>,
    ) -> Substitution {
        let tys = parameters.iter(&Interner).map(|p| from_chalk(db, p.clone())).collect();
        Substitution::intern(tys)
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef<Interner>;

    fn to_chalk(self: TraitRef, db: &dyn HirDatabase) -> chalk_ir::TraitRef<Interner> {
        let trait_id = self.trait_id;
        let substitution = self.substitution.to_chalk(db);
        chalk_ir::TraitRef { trait_id, substitution }
    }

    fn from_chalk(db: &dyn HirDatabase, trait_ref: chalk_ir::TraitRef<Interner>) -> Self {
        let trait_id = trait_ref.trait_id;
        let substs = from_chalk(db, trait_ref.substitution);
        TraitRef { trait_id, substitution: substs }
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
        match self {
            WhereClause::Implemented(trait_ref) => {
                chalk_ir::WhereClause::Implemented(trait_ref.to_chalk(db))
            }
            WhereClause::AliasEq(alias_eq) => chalk_ir::WhereClause::AliasEq(alias_eq.to_chalk(db)),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        where_clause: chalk_ir::WhereClause<Interner>,
    ) -> WhereClause {
        match where_clause {
            chalk_ir::WhereClause::Implemented(tr) => WhereClause::Implemented(from_chalk(db, tr)),
            chalk_ir::WhereClause::AliasEq(alias_eq) => {
                WhereClause::AliasEq(from_chalk(db, alias_eq))
            }

            chalk_ir::WhereClause::LifetimeOutlives(_) => {
                // we shouldn't get these from Chalk
                panic!("encountered LifetimeOutlives from Chalk")
            }

            chalk_ir::WhereClause::TypeOutlives(_) => {
                // we shouldn't get these from Chalk
                panic!("encountered TypeOutlives from Chalk")
            }
        }
    }
}

impl ToChalk for ProjectionTy {
    type Chalk = chalk_ir::ProjectionTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::ProjectionTy<Interner> {
        chalk_ir::ProjectionTy {
            associated_ty_id: self.associated_ty_id,
            substitution: self.substitution.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<Interner>,
    ) -> ProjectionTy {
        ProjectionTy {
            associated_ty_id: projection_ty.associated_ty_id,
            substitution: from_chalk(db, projection_ty.substitution),
        }
    }
}
impl ToChalk for OpaqueTy {
    type Chalk = chalk_ir::OpaqueTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk {
        chalk_ir::OpaqueTy {
            opaque_ty_id: self.opaque_ty_id,
            substitution: self.substitution.to_chalk(db),
        }
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        OpaqueTy {
            opaque_ty_id: chalk.opaque_ty_id,
            substitution: from_chalk(db, chalk.substitution),
        }
    }
}

impl ToChalk for AliasTy {
    type Chalk = chalk_ir::AliasTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk {
        match self {
            AliasTy::Projection(projection_ty) => {
                chalk_ir::AliasTy::Projection(projection_ty.to_chalk(db))
            }
            AliasTy::Opaque(opaque_ty) => chalk_ir::AliasTy::Opaque(opaque_ty.to_chalk(db)),
        }
    }

    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self {
        match chalk {
            chalk_ir::AliasTy::Projection(projection_ty) => {
                AliasTy::Projection(from_chalk(db, projection_ty))
            }
            chalk_ir::AliasTy::Opaque(opaque_ty) => AliasTy::Opaque(from_chalk(db, opaque_ty)),
        }
    }
}

impl ToChalk for AliasEq {
    type Chalk = chalk_ir::AliasEq<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::AliasEq<Interner> {
        chalk_ir::AliasEq { alias: self.alias.to_chalk(db), ty: self.ty.to_chalk(db) }
    }

    fn from_chalk(db: &dyn HirDatabase, alias_eq: chalk_ir::AliasEq<Interner>) -> Self {
        AliasEq { alias: from_chalk(db, alias_eq.alias), ty: from_chalk(db, alias_eq.ty) }
    }
}

impl ToChalk for DomainGoal {
    type Chalk = chalk_ir::DomainGoal<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::DomainGoal<Interner> {
        match self {
            DomainGoal::Holds(WhereClause::Implemented(tr)) => tr.to_chalk(db).cast(&Interner),
            DomainGoal::Holds(WhereClause::AliasEq(alias_eq)) => {
                alias_eq.to_chalk(db).cast(&Interner)
            }
        }
    }

    fn from_chalk(_db: &dyn HirDatabase, _goal: chalk_ir::DomainGoal<Interner>) -> Self {
        unimplemented!()
    }
}

impl<T> ToChalk for Canonical<T>
where
    T: ToChalk,
    T::Chalk: HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::Canonical<T::Chalk>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Canonical<T::Chalk> {
        let value = self.value.to_chalk(db);
        chalk_ir::Canonical { value, binders: self.binders }
    }

    fn from_chalk(db: &dyn HirDatabase, canonical: chalk_ir::Canonical<T::Chalk>) -> Canonical<T> {
        Canonical { binders: canonical.binders, value: from_chalk(db, canonical.value) }
    }
}

impl<T: ToChalk> ToChalk for InEnvironment<T>
where
    T::Chalk: chalk_ir::interner::HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::InEnvironment<T::Chalk>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::InEnvironment<T::Chalk> {
        chalk_ir::InEnvironment { environment: self.environment, goal: self.goal.to_chalk(db) }
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        _in_env: chalk_ir::InEnvironment<T::Chalk>,
    ) -> InEnvironment<T> {
        unimplemented!()
    }
}

impl<T: ToChalk> ToChalk for crate::Binders<T>
where
    T::Chalk: chalk_ir::interner::HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::Binders<T::Chalk>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Binders<T::Chalk> {
        let (value, binders) = self.into_value_and_skipped_binders();
        chalk_ir::Binders::new(binders, value.to_chalk(db))
    }

    fn from_chalk(db: &dyn HirDatabase, binders: chalk_ir::Binders<T::Chalk>) -> crate::Binders<T> {
        let (v, b) = binders.into_value_and_skipped_binders();
        crate::Binders::new(b, from_chalk(db, v))
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
    let self_ty_shifted_in = self_ty.clone().shifted_in_from(DebruijnIndex::ONE);
    let (pred, binders) = pred.as_ref().into_value_and_skipped_binders();
    match pred {
        WhereClause::Implemented(trait_ref) => {
            if trait_ref.self_type_parameter(&Interner) != &self_ty_shifted_in {
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
            if projection_ty.self_type_parameter(&Interner) != &self_ty_shifted_in {
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
