//! Conversion code from/to Chalk.
use std::{fmt, sync::Arc};

use log::debug;

use chalk_ir::{
    cast::Cast, fold::shift::Shift, interner::HasInterner, GenericArg, Goal, GoalData,
    PlaceholderIndex, TypeName, UniverseIndex,
};

use hir_def::{AssocContainerId, AssocItemId, GenericDefId, HasModule, Lookup, TypeAliasId};
use ra_db::{
    salsa::{InternId, InternKey},
    CrateId,
};

use super::{builtin, AssocTyValue, Canonical, ChalkContext, Impl, Obligation};
use crate::{
    db::HirDatabase, display::HirDisplay, method_resolution::TyFingerprint, utils::generics,
    ApplicationTy, DebruijnIndex, GenericPredicate, ProjectionTy, Substs, TraitRef, Ty, TypeCtor,
};

pub(super) mod tls;

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Interner;

impl chalk_ir::interner::Interner for Interner {
    type InternedType = Box<chalk_ir::TyData<Self>>; // FIXME use Arc?
    type InternedLifetime = chalk_ir::LifetimeData<Self>;
    type InternedConst = Arc<chalk_ir::ConstData<Self>>;
    type InternedConcreteConst = ();
    type InternedGenericArg = chalk_ir::GenericArgData<Self>;
    type InternedGoal = Arc<GoalData<Self>>;
    type InternedGoals = Vec<Goal<Self>>;
    type InternedSubstitution = Vec<GenericArg<Self>>;
    type InternedProgramClause = chalk_ir::ProgramClauseData<Self>;
    type InternedProgramClauses = Arc<[chalk_ir::ProgramClause<Self>]>;
    type InternedQuantifiedWhereClauses = Vec<chalk_ir::QuantifiedWhereClause<Self>>;
    type InternedVariableKinds = Vec<chalk_ir::VariableKind<Self>>;
    type InternedCanonicalVarKinds = Vec<chalk_ir::CanonicalVarKind<Self>>;
    type DefId = InternId;
    type InternedAdtId = InternId;
    type Identifier = TypeAliasId;

    fn debug_adt_id(type_kind_id: StructId, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_struct_id(type_kind_id, fmt)))
    }

    fn debug_trait_id(type_kind_id: TraitId, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_trait_id(type_kind_id, fmt)))
    }

    fn debug_assoc_type_id(id: AssocTypeId, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_assoc_type_id(id, fmt)))
    }

    fn debug_alias(
        alias: &chalk_ir::AliasTy<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_alias(alias, fmt)))
    }

    fn debug_projection_ty(
        proj: &chalk_ir::ProjectionTy<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_projection_ty(proj, fmt)))
    }

    fn debug_opaque_ty(
        opaque_ty: &chalk_ir::OpaqueTy<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_opaque_ty(opaque_ty, fmt)))
    }

    fn debug_opaque_ty_id(
        opaque_ty_id: chalk_ir::OpaqueTyId<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_opaque_ty_id(opaque_ty_id, fmt)))
    }

    fn debug_ty(ty: &chalk_ir::Ty<Interner>, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_ty(ty, fmt)))
    }

    fn debug_lifetime(
        lifetime: &chalk_ir::Lifetime<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_lifetime(lifetime, fmt)))
    }

    fn debug_generic_arg(
        parameter: &GenericArg<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_generic_arg(parameter, fmt)))
    }

    fn debug_goal(goal: &Goal<Interner>, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_goal(goal, fmt)))
    }

    fn debug_goals(
        goals: &chalk_ir::Goals<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_goals(goals, fmt)))
    }

    fn debug_program_clause_implication(
        pci: &chalk_ir::ProgramClauseImplication<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_program_clause_implication(pci, fmt)))
    }

    fn debug_application_ty(
        application_ty: &chalk_ir::ApplicationTy<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_application_ty(application_ty, fmt)))
    }

    fn debug_substitution(
        substitution: &chalk_ir::Substitution<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_substitution(substitution, fmt)))
    }

    fn debug_separator_trait_ref(
        separator_trait_ref: &chalk_ir::SeparatorTraitRef<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| {
            Some(prog?.debug_separator_trait_ref(separator_trait_ref, fmt))
        })
    }

    fn intern_ty(&self, ty: chalk_ir::TyData<Self>) -> Box<chalk_ir::TyData<Self>> {
        Box::new(ty)
    }

    fn ty_data<'a>(&self, ty: &'a Box<chalk_ir::TyData<Self>>) -> &'a chalk_ir::TyData<Self> {
        ty
    }

    fn intern_lifetime(
        &self,
        lifetime: chalk_ir::LifetimeData<Self>,
    ) -> chalk_ir::LifetimeData<Self> {
        lifetime
    }

    fn lifetime_data<'a>(
        &self,
        lifetime: &'a chalk_ir::LifetimeData<Self>,
    ) -> &'a chalk_ir::LifetimeData<Self> {
        lifetime
    }

    fn intern_const(&self, constant: chalk_ir::ConstData<Self>) -> Arc<chalk_ir::ConstData<Self>> {
        Arc::new(constant)
    }

    fn const_data<'a>(
        &self,
        constant: &'a Arc<chalk_ir::ConstData<Self>>,
    ) -> &'a chalk_ir::ConstData<Self> {
        constant
    }

    fn const_eq(&self, _ty: &Box<chalk_ir::TyData<Self>>, _c1: &(), _c2: &()) -> bool {
        true
    }

    fn intern_generic_arg(
        &self,
        parameter: chalk_ir::GenericArgData<Self>,
    ) -> chalk_ir::GenericArgData<Self> {
        parameter
    }

    fn generic_arg_data<'a>(
        &self,
        parameter: &'a chalk_ir::GenericArgData<Self>,
    ) -> &'a chalk_ir::GenericArgData<Self> {
        parameter
    }

    fn intern_goal(&self, goal: GoalData<Self>) -> Arc<GoalData<Self>> {
        Arc::new(goal)
    }

    fn intern_goals<E>(
        &self,
        data: impl IntoIterator<Item = Result<Goal<Self>, E>>,
    ) -> Result<Self::InternedGoals, E> {
        data.into_iter().collect()
    }

    fn goal_data<'a>(&self, goal: &'a Arc<GoalData<Self>>) -> &'a GoalData<Self> {
        goal
    }

    fn goals_data<'a>(&self, goals: &'a Vec<Goal<Interner>>) -> &'a [Goal<Interner>] {
        goals
    }

    fn intern_substitution<E>(
        &self,
        data: impl IntoIterator<Item = Result<GenericArg<Self>, E>>,
    ) -> Result<Vec<GenericArg<Self>>, E> {
        data.into_iter().collect()
    }

    fn substitution_data<'a>(
        &self,
        substitution: &'a Vec<GenericArg<Self>>,
    ) -> &'a [GenericArg<Self>] {
        substitution
    }

    fn intern_program_clause(
        &self,
        data: chalk_ir::ProgramClauseData<Self>,
    ) -> chalk_ir::ProgramClauseData<Self> {
        data
    }

    fn program_clause_data<'a>(
        &self,
        clause: &'a chalk_ir::ProgramClauseData<Self>,
    ) -> &'a chalk_ir::ProgramClauseData<Self> {
        clause
    }

    fn intern_program_clauses<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::ProgramClause<Self>, E>>,
    ) -> Result<Arc<[chalk_ir::ProgramClause<Self>]>, E> {
        data.into_iter().collect()
    }

    fn program_clauses_data<'a>(
        &self,
        clauses: &'a Arc<[chalk_ir::ProgramClause<Self>]>,
    ) -> &'a [chalk_ir::ProgramClause<Self>] {
        &clauses
    }

    fn intern_quantified_where_clauses<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::QuantifiedWhereClause<Self>, E>>,
    ) -> Result<Self::InternedQuantifiedWhereClauses, E> {
        data.into_iter().collect()
    }

    fn quantified_where_clauses_data<'a>(
        &self,
        clauses: &'a Self::InternedQuantifiedWhereClauses,
    ) -> &'a [chalk_ir::QuantifiedWhereClause<Self>] {
        clauses
    }

    fn intern_generic_arg_kinds<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::VariableKind<Self>, E>>,
    ) -> Result<Self::InternedVariableKinds, E> {
        data.into_iter().collect()
    }

    fn variable_kinds_data<'a>(
        &self,
        parameter_kinds: &'a Self::InternedVariableKinds,
    ) -> &'a [chalk_ir::VariableKind<Self>] {
        &parameter_kinds
    }

    fn intern_canonical_var_kinds<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::CanonicalVarKind<Self>, E>>,
    ) -> Result<Self::InternedCanonicalVarKinds, E> {
        data.into_iter().collect()
    }

    fn canonical_var_kinds_data<'a>(
        &self,
        canonical_var_kinds: &'a Self::InternedCanonicalVarKinds,
    ) -> &'a [chalk_ir::CanonicalVarKind<Self>] {
        &canonical_var_kinds
    }
}

impl chalk_ir::interner::HasInterner for Interner {
    type Interner = Self;
}

pub type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub type AssociatedTyDatum = chalk_rust_ir::AssociatedTyDatum<Interner>;
pub type TraitId = chalk_ir::TraitId<Interner>;
pub type TraitDatum = chalk_rust_ir::TraitDatum<Interner>;
pub type StructId = chalk_ir::AdtId<Interner>;
pub type StructDatum = chalk_rust_ir::AdtDatum<Interner>;
pub type ImplId = chalk_ir::ImplId<Interner>;
pub type ImplDatum = chalk_rust_ir::ImplDatum<Interner>;
pub type AssociatedTyValueId = chalk_rust_ir::AssociatedTyValueId<Interner>;
pub type AssociatedTyValue = chalk_rust_ir::AssociatedTyValue<Interner>;

pub(super) trait ToChalk {
    type Chalk;
    fn to_chalk(self, db: &dyn HirDatabase) -> Self::Chalk;
    fn from_chalk(db: &dyn HirDatabase, chalk: Self::Chalk) -> Self;
}

pub(super) fn from_chalk<T, ChalkT>(db: &dyn HirDatabase, chalk: ChalkT) -> T
where
    T: ToChalk<Chalk = ChalkT>,
{
    T::from_chalk(db, chalk)
}

impl ToChalk for Ty {
    type Chalk = chalk_ir::Ty<Interner>;
    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Ty<Interner> {
        match self {
            Ty::Apply(apply_ty) => {
                let name = apply_ty.ctor.to_chalk(db);
                let substitution = apply_ty.parameters.to_chalk(db);
                chalk_ir::ApplicationTy { name, substitution }.cast(&Interner).intern(&Interner)
            }
            Ty::Projection(proj_ty) => {
                let associated_ty_id = proj_ty.associated_ty.to_chalk(db);
                let substitution = proj_ty.parameters.to_chalk(db);
                chalk_ir::AliasTy::Projection(chalk_ir::ProjectionTy {
                    associated_ty_id,
                    substitution,
                })
                .cast(&Interner)
                .intern(&Interner)
            }
            Ty::Placeholder(id) => {
                let interned_id = db.intern_type_param_id(id);
                PlaceholderIndex {
                    ui: UniverseIndex::ROOT,
                    idx: interned_id.as_intern_id().as_usize(),
                }
                .to_ty::<Interner>(&Interner)
            }
            Ty::Bound(idx) => chalk_ir::TyData::BoundVar(idx).intern(&Interner),
            Ty::Infer(_infer_ty) => panic!("uncanonicalized infer ty"),
            Ty::Dyn(predicates) => {
                let where_clauses = chalk_ir::QuantifiedWhereClauses::from(
                    &Interner,
                    predicates.iter().filter(|p| !p.is_error()).cloned().map(|p| p.to_chalk(db)),
                );
                let bounded_ty = chalk_ir::DynTy { bounds: make_binders(where_clauses, 1) };
                chalk_ir::TyData::Dyn(bounded_ty).intern(&Interner)
            }
            Ty::Opaque(_) | Ty::Unknown => {
                let substitution = chalk_ir::Substitution::empty(&Interner);
                let name = TypeName::Error;
                chalk_ir::ApplicationTy { name, substitution }.cast(&Interner).intern(&Interner)
            }
        }
    }
    fn from_chalk(db: &dyn HirDatabase, chalk: chalk_ir::Ty<Interner>) -> Self {
        match chalk.data(&Interner).clone() {
            chalk_ir::TyData::Apply(apply_ty) => match apply_ty.name {
                TypeName::Error => Ty::Unknown,
                _ => {
                    let ctor = from_chalk(db, apply_ty.name);
                    let parameters = from_chalk(db, apply_ty.substitution);
                    Ty::Apply(ApplicationTy { ctor, parameters })
                }
            },
            chalk_ir::TyData::Placeholder(idx) => {
                assert_eq!(idx.ui, UniverseIndex::ROOT);
                let interned_id = crate::db::GlobalTypeParamId::from_intern_id(
                    crate::salsa::InternId::from(idx.idx),
                );
                Ty::Placeholder(db.lookup_intern_type_param_id(interned_id))
            }
            chalk_ir::TyData::Alias(chalk_ir::AliasTy::Projection(proj)) => {
                let associated_ty = from_chalk(db, proj.associated_ty_id);
                let parameters = from_chalk(db, proj.substitution);
                Ty::Projection(ProjectionTy { associated_ty, parameters })
            }
            chalk_ir::TyData::Alias(chalk_ir::AliasTy::Opaque(_)) => unimplemented!(),
            chalk_ir::TyData::Function(_) => unimplemented!(),
            chalk_ir::TyData::BoundVar(idx) => Ty::Bound(idx),
            chalk_ir::TyData::InferenceVar(_iv) => Ty::Unknown,
            chalk_ir::TyData::Dyn(where_clauses) => {
                assert_eq!(where_clauses.bounds.binders.len(&Interner), 1);
                let predicates = where_clauses
                    .bounds
                    .skip_binders()
                    .iter(&Interner)
                    .map(|c| from_chalk(db, c.clone()))
                    .collect();
                Ty::Dyn(predicates)
            }
        }
    }
}

impl ToChalk for Substs {
    type Chalk = chalk_ir::Substitution<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Substitution<Interner> {
        chalk_ir::Substitution::from(&Interner, self.iter().map(|ty| ty.clone().to_chalk(db)))
    }

    fn from_chalk(db: &dyn HirDatabase, parameters: chalk_ir::Substitution<Interner>) -> Substs {
        let tys = parameters
            .iter(&Interner)
            .map(|p| match p.ty(&Interner) {
                Some(ty) => from_chalk(db, ty.clone()),
                None => unimplemented!(),
            })
            .collect();
        Substs(tys)
    }
}

impl ToChalk for TraitRef {
    type Chalk = chalk_ir::TraitRef<Interner>;

    fn to_chalk(self: TraitRef, db: &dyn HirDatabase) -> chalk_ir::TraitRef<Interner> {
        let trait_id = self.trait_.to_chalk(db);
        let substitution = self.substs.to_chalk(db);
        chalk_ir::TraitRef { trait_id, substitution }
    }

    fn from_chalk(db: &dyn HirDatabase, trait_ref: chalk_ir::TraitRef<Interner>) -> Self {
        let trait_ = from_chalk(db, trait_ref.trait_id);
        let substs = from_chalk(db, trait_ref.substitution);
        TraitRef { trait_, substs }
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

impl ToChalk for TypeCtor {
    type Chalk = TypeName<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> TypeName<Interner> {
        match self {
            TypeCtor::AssociatedType(type_alias) => {
                let type_id = type_alias.to_chalk(db);
                TypeName::AssociatedType(type_id)
            }
            _ => {
                // other TypeCtors get interned and turned into a chalk StructId
                let struct_id = db.intern_type_ctor(self).into();
                TypeName::Adt(struct_id)
            }
        }
    }

    fn from_chalk(db: &dyn HirDatabase, type_name: TypeName<Interner>) -> TypeCtor {
        match type_name {
            TypeName::Adt(struct_id) => db.lookup_intern_type_ctor(struct_id.into()),
            TypeName::AssociatedType(type_id) => TypeCtor::AssociatedType(from_chalk(db, type_id)),
            TypeName::OpaqueType(_) => unreachable!(),

            TypeName::Scalar(_) => unreachable!(),
            TypeName::Tuple(_) => unreachable!(),
            TypeName::Raw(_) => unreachable!(),
            TypeName::Slice => unreachable!(),
            TypeName::Ref(_) => unreachable!(),
            TypeName::Str => unreachable!(),

            TypeName::FnDef(_) => unreachable!(),

            TypeName::Error => {
                // this should not be reached, since we don't represent TypeName::Error with TypeCtor
                unreachable!()
            }
        }
    }
}

impl ToChalk for Impl {
    type Chalk = ImplId;

    fn to_chalk(self, db: &dyn HirDatabase) -> ImplId {
        db.intern_chalk_impl(self).into()
    }

    fn from_chalk(db: &dyn HirDatabase, impl_id: ImplId) -> Impl {
        db.lookup_intern_chalk_impl(impl_id.into())
    }
}

impl ToChalk for TypeAliasId {
    type Chalk = AssocTypeId;

    fn to_chalk(self, _db: &dyn HirDatabase) -> AssocTypeId {
        chalk_ir::AssocTypeId(self.as_intern_id())
    }

    fn from_chalk(_db: &dyn HirDatabase, type_alias_id: AssocTypeId) -> TypeAliasId {
        InternKey::from_intern_id(type_alias_id.0)
    }
}

impl ToChalk for AssocTyValue {
    type Chalk = AssociatedTyValueId;

    fn to_chalk(self, db: &dyn HirDatabase) -> AssociatedTyValueId {
        db.intern_assoc_ty_value(self).into()
    }

    fn from_chalk(db: &dyn HirDatabase, assoc_ty_value_id: AssociatedTyValueId) -> AssocTyValue {
        db.lookup_intern_assoc_ty_value(assoc_ty_value_id.into())
    }
}

impl ToChalk for GenericPredicate {
    type Chalk = chalk_ir::QuantifiedWhereClause<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::QuantifiedWhereClause<Interner> {
        match self {
            GenericPredicate::Implemented(trait_ref) => {
                let chalk_trait_ref = trait_ref.to_chalk(db);
                let chalk_trait_ref = chalk_trait_ref.shifted_in(&Interner);
                make_binders(chalk_ir::WhereClause::Implemented(chalk_trait_ref), 0)
            }
            GenericPredicate::Projection(projection_pred) => {
                let ty = projection_pred.ty.to_chalk(db).shifted_in(&Interner);
                let projection = projection_pred.projection_ty.to_chalk(db).shifted_in(&Interner);
                let alias = chalk_ir::AliasTy::Projection(projection);
                make_binders(chalk_ir::WhereClause::AliasEq(chalk_ir::AliasEq { alias, ty }), 0)
            }
            GenericPredicate::Error => panic!("tried passing GenericPredicate::Error to Chalk"),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        where_clause: chalk_ir::QuantifiedWhereClause<Interner>,
    ) -> GenericPredicate {
        // we don't produce any where clauses with binders and can't currently deal with them
        match where_clause
            .skip_binders()
            .shifted_out(&Interner)
            .expect("unexpected bound vars in where clause")
        {
            chalk_ir::WhereClause::Implemented(tr) => {
                GenericPredicate::Implemented(from_chalk(db, tr))
            }
            chalk_ir::WhereClause::AliasEq(projection_eq) => {
                let projection_ty = from_chalk(
                    db,
                    match projection_eq.alias {
                        chalk_ir::AliasTy::Projection(p) => p,
                        _ => unimplemented!(),
                    },
                );
                let ty = from_chalk(db, projection_eq.ty);
                GenericPredicate::Projection(super::ProjectionPredicate { projection_ty, ty })
            }
        }
    }
}

impl ToChalk for ProjectionTy {
    type Chalk = chalk_ir::ProjectionTy<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::ProjectionTy<Interner> {
        chalk_ir::ProjectionTy {
            associated_ty_id: self.associated_ty.to_chalk(db),
            substitution: self.parameters.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        projection_ty: chalk_ir::ProjectionTy<Interner>,
    ) -> ProjectionTy {
        ProjectionTy {
            associated_ty: from_chalk(db, projection_ty.associated_ty_id),
            parameters: from_chalk(db, projection_ty.substitution),
        }
    }
}

impl ToChalk for super::ProjectionPredicate {
    type Chalk = chalk_ir::AliasEq<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::AliasEq<Interner> {
        chalk_ir::AliasEq {
            alias: chalk_ir::AliasTy::Projection(self.projection_ty.to_chalk(db)),
            ty: self.ty.to_chalk(db),
        }
    }

    fn from_chalk(_db: &dyn HirDatabase, _normalize: chalk_ir::AliasEq<Interner>) -> Self {
        unimplemented!()
    }
}

impl ToChalk for Obligation {
    type Chalk = chalk_ir::DomainGoal<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::DomainGoal<Interner> {
        match self {
            Obligation::Trait(tr) => tr.to_chalk(db).cast(&Interner),
            Obligation::Projection(pr) => pr.to_chalk(db).cast(&Interner),
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
        let parameter = chalk_ir::CanonicalVarKind::new(
            chalk_ir::VariableKind::Ty,
            chalk_ir::UniverseIndex::ROOT,
        );
        let value = self.value.to_chalk(db);
        chalk_ir::Canonical {
            value,
            binders: chalk_ir::CanonicalVarKinds::from(&Interner, vec![parameter; self.num_vars]),
        }
    }

    fn from_chalk(db: &dyn HirDatabase, canonical: chalk_ir::Canonical<T::Chalk>) -> Canonical<T> {
        Canonical {
            num_vars: canonical.binders.len(&Interner),
            value: from_chalk(db, canonical.value),
        }
    }
}

impl ToChalk for Arc<super::TraitEnvironment> {
    type Chalk = chalk_ir::Environment<Interner>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::Environment<Interner> {
        let mut clauses = Vec::new();
        for pred in &self.predicates {
            if pred.is_error() {
                // for env, we just ignore errors
                continue;
            }
            let program_clause: chalk_ir::ProgramClause<Interner> =
                pred.clone().to_chalk(db).cast(&Interner);
            clauses.push(program_clause.into_from_env_clause(&Interner));
        }
        chalk_ir::Environment::new(&Interner).add_clauses(&Interner, clauses)
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        _env: chalk_ir::Environment<Interner>,
    ) -> Arc<super::TraitEnvironment> {
        unimplemented!()
    }
}

impl<T: ToChalk> ToChalk for super::InEnvironment<T>
where
    T::Chalk: chalk_ir::interner::HasInterner<Interner = Interner>,
{
    type Chalk = chalk_ir::InEnvironment<T::Chalk>;

    fn to_chalk(self, db: &dyn HirDatabase) -> chalk_ir::InEnvironment<T::Chalk> {
        chalk_ir::InEnvironment {
            environment: self.environment.to_chalk(db),
            goal: self.value.to_chalk(db),
        }
    }

    fn from_chalk(
        db: &dyn HirDatabase,
        in_env: chalk_ir::InEnvironment<T::Chalk>,
    ) -> super::InEnvironment<T> {
        super::InEnvironment {
            environment: from_chalk(db, in_env.environment),
            value: from_chalk(db, in_env.goal),
        }
    }
}

impl ToChalk for builtin::BuiltinImplData {
    type Chalk = ImplDatum;

    fn to_chalk(self, db: &dyn HirDatabase) -> ImplDatum {
        let impl_type = chalk_rust_ir::ImplType::External;
        let where_clauses = self.where_clauses.into_iter().map(|w| w.to_chalk(db)).collect();

        let impl_datum_bound =
            chalk_rust_ir::ImplDatumBound { trait_ref: self.trait_ref.to_chalk(db), where_clauses };
        let associated_ty_value_ids =
            self.assoc_ty_values.into_iter().map(|v| v.to_chalk(db)).collect();
        chalk_rust_ir::ImplDatum {
            binders: make_binders(impl_datum_bound, self.num_vars),
            impl_type,
            polarity: chalk_rust_ir::Polarity::Positive,
            associated_ty_value_ids,
        }
    }

    fn from_chalk(_db: &dyn HirDatabase, _data: ImplDatum) -> Self {
        unimplemented!()
    }
}

impl ToChalk for builtin::BuiltinImplAssocTyValueData {
    type Chalk = AssociatedTyValue;

    fn to_chalk(self, db: &dyn HirDatabase) -> AssociatedTyValue {
        let ty = self.value.to_chalk(db);
        let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty };

        chalk_rust_ir::AssociatedTyValue {
            associated_ty_id: self.assoc_ty_id.to_chalk(db),
            impl_id: self.impl_.to_chalk(db),
            value: make_binders(value_bound, self.num_vars),
        }
    }

    fn from_chalk(
        _db: &dyn HirDatabase,
        _data: AssociatedTyValue,
    ) -> builtin::BuiltinImplAssocTyValueData {
        unimplemented!()
    }
}

fn make_binders<T>(value: T, num_vars: usize) -> chalk_ir::Binders<T>
where
    T: HasInterner<Interner = Interner>,
{
    chalk_ir::Binders::new(
        chalk_ir::VariableKinds::from(
            &Interner,
            std::iter::repeat(chalk_ir::VariableKind::Ty).take(num_vars),
        ),
        value,
    )
}

fn convert_where_clauses(
    db: &dyn HirDatabase,
    def: GenericDefId,
    substs: &Substs,
) -> Vec<chalk_ir::QuantifiedWhereClause<Interner>> {
    let generic_predicates = db.generic_predicates(def);
    let mut result = Vec::with_capacity(generic_predicates.len());
    for pred in generic_predicates.iter() {
        if pred.value.is_error() {
            // skip errored predicates completely
            continue;
        }
        result.push(pred.clone().subst(substs).to_chalk(db));
    }
    result
}

fn generic_predicate_to_inline_bound(
    db: &dyn HirDatabase,
    pred: &GenericPredicate,
    self_ty: &Ty,
) -> Option<chalk_rust_ir::InlineBound<Interner>> {
    // An InlineBound is like a GenericPredicate, except the self type is left out.
    // We don't have a special type for this, but Chalk does.
    match pred {
        GenericPredicate::Implemented(trait_ref) => {
            if &trait_ref.substs[0] != self_ty {
                // we can only convert predicates back to type bounds if they
                // have the expected self type
                return None;
            }
            let args_no_self = trait_ref.substs[1..]
                .iter()
                .map(|ty| ty.clone().to_chalk(db).cast(&Interner))
                .collect();
            let trait_bound =
                chalk_rust_ir::TraitBound { trait_id: trait_ref.trait_.to_chalk(db), args_no_self };
            Some(chalk_rust_ir::InlineBound::TraitBound(trait_bound))
        }
        GenericPredicate::Projection(proj) => {
            if &proj.projection_ty.parameters[0] != self_ty {
                return None;
            }
            let trait_ = match proj.projection_ty.associated_ty.lookup(db.upcast()).container {
                AssocContainerId::TraitId(t) => t,
                _ => panic!("associated type not in trait"),
            };
            let args_no_self = proj.projection_ty.parameters[1..]
                .iter()
                .map(|ty| ty.clone().to_chalk(db).cast(&Interner))
                .collect();
            let alias_eq_bound = chalk_rust_ir::AliasEqBound {
                value: proj.ty.clone().to_chalk(db),
                trait_bound: chalk_rust_ir::TraitBound {
                    trait_id: trait_.to_chalk(db),
                    args_no_self,
                },
                associated_ty_id: proj.projection_ty.associated_ty.to_chalk(db),
                parameters: Vec::new(), // FIXME we don't support generic associated types yet
            };
            Some(chalk_rust_ir::InlineBound::AliasEqBound(alias_eq_bound))
        }
        GenericPredicate::Error => None,
    }
}

impl<'a> chalk_solve::RustIrDatabase<Interner> for ChalkContext<'a> {
    fn associated_ty_data(&self, id: AssocTypeId) -> Arc<AssociatedTyDatum> {
        self.db.associated_ty_data(id)
    }
    fn trait_datum(&self, trait_id: TraitId) -> Arc<TraitDatum> {
        self.db.trait_datum(self.krate, trait_id)
    }
    fn adt_datum(&self, struct_id: StructId) -> Arc<StructDatum> {
        self.db.struct_datum(self.krate, struct_id)
    }
    fn impl_datum(&self, impl_id: ImplId) -> Arc<ImplDatum> {
        self.db.impl_datum(self.krate, impl_id)
    }

    fn fn_def_datum(
        &self,
        _fn_def_id: chalk_ir::FnDefId<Interner>,
    ) -> Arc<chalk_rust_ir::FnDefDatum<Interner>> {
        // We don't yet provide any FnDefs to Chalk
        unimplemented!()
    }

    fn impls_for_trait(
        &self,
        trait_id: TraitId,
        parameters: &[GenericArg<Interner>],
    ) -> Vec<ImplId> {
        debug!("impls_for_trait {:?}", trait_id);
        let trait_: hir_def::TraitId = from_chalk(self.db, trait_id);

        let ty: Ty = from_chalk(self.db, parameters[0].assert_ty_ref(&Interner).clone());

        let self_ty_fp = TyFingerprint::for_impl(&ty);

        // Note: Since we're using impls_for_trait, only impls where the trait
        // can be resolved should ever reach Chalk. `impl_datum` relies on that
        // and will panic if the trait can't be resolved.
        let mut result: Vec<_> = self
            .db
            .impls_for_trait(self.krate, trait_, self_ty_fp)
            .iter()
            .copied()
            .map(Impl::ImplDef)
            .map(|impl_| impl_.to_chalk(self.db))
            .collect();

        let arg: Option<Ty> =
            parameters.get(1).map(|p| from_chalk(self.db, p.assert_ty_ref(&Interner).clone()));

        builtin::get_builtin_impls(self.db, self.krate, &ty, &arg, trait_, |i| {
            result.push(i.to_chalk(self.db))
        });

        debug!("impls_for_trait returned {} impls", result.len());
        result
    }
    fn impl_provided_for(&self, auto_trait_id: TraitId, struct_id: StructId) -> bool {
        debug!("impl_provided_for {:?}, {:?}", auto_trait_id, struct_id);
        false // FIXME
    }
    fn associated_ty_value(&self, id: AssociatedTyValueId) -> Arc<AssociatedTyValue> {
        self.db.associated_ty_value(self.krate, id)
    }
    fn custom_clauses(&self) -> Vec<chalk_ir::ProgramClause<Interner>> {
        vec![]
    }
    fn local_impls_to_coherence_check(&self, _trait_id: TraitId) -> Vec<ImplId> {
        // We don't do coherence checking (yet)
        unimplemented!()
    }
    fn interner(&self) -> &Interner {
        &Interner
    }
    fn well_known_trait_id(
        &self,
        _well_known_trait: chalk_rust_ir::WellKnownTrait,
    ) -> Option<chalk_ir::TraitId<Interner>> {
        // FIXME tell Chalk about well-known traits (here and in trait_datum)
        None
    }

    fn program_clauses_for_env(
        &self,
        environment: &chalk_ir::Environment<Interner>,
    ) -> chalk_ir::ProgramClauses<Interner> {
        self.db.program_clauses_for_chalk_env(self.krate, environment.clone())
    }

    fn opaque_ty_data(
        &self,
        _id: chalk_ir::OpaqueTyId<Interner>,
    ) -> Arc<chalk_rust_ir::OpaqueTyDatum<Interner>> {
        unimplemented!()
    }

    fn force_impl_for(
        &self,
        _well_known: chalk_rust_ir::WellKnownTrait,
        _ty: &chalk_ir::TyData<Interner>,
    ) -> Option<bool> {
        // this method is mostly for rustc
        None
    }

    fn is_object_safe(&self, _trait_id: chalk_ir::TraitId<Interner>) -> bool {
        // FIXME: implement actual object safety
        true
    }
}

pub(crate) fn program_clauses_for_chalk_env_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    environment: chalk_ir::Environment<Interner>,
) -> chalk_ir::ProgramClauses<Interner> {
    chalk_solve::program_clauses_for_env(&ChalkContext { db, krate }, &environment)
}

pub(crate) fn associated_ty_data_query(
    db: &dyn HirDatabase,
    id: AssocTypeId,
) -> Arc<AssociatedTyDatum> {
    debug!("associated_ty_data {:?}", id);
    let type_alias: TypeAliasId = from_chalk(db, id);
    let trait_ = match type_alias.lookup(db.upcast()).container {
        AssocContainerId::TraitId(t) => t,
        _ => panic!("associated type not in trait"),
    };

    // Lower bounds -- we could/should maybe move this to a separate query in `lower`
    let type_alias_data = db.type_alias_data(type_alias);
    let generic_params = generics(db.upcast(), type_alias.into());
    let bound_vars = Substs::bound_vars(&generic_params, DebruijnIndex::INNERMOST);
    let resolver = hir_def::resolver::HasResolver::resolver(type_alias, db.upcast());
    let ctx = crate::TyLoweringContext::new(db, &resolver)
        .with_type_param_mode(crate::lower::TypeParamLoweringMode::Variable);
    let self_ty = Ty::Bound(crate::BoundVar::new(crate::DebruijnIndex::INNERMOST, 0));
    let bounds = type_alias_data
        .bounds
        .iter()
        .flat_map(|bound| GenericPredicate::from_type_bound(&ctx, bound, self_ty.clone()))
        .filter_map(|pred| generic_predicate_to_inline_bound(db, &pred, &self_ty))
        .map(|bound| make_binders(bound.shifted_in(&Interner), 0))
        .collect();

    let where_clauses = convert_where_clauses(db, type_alias.into(), &bound_vars);
    let bound_data = chalk_rust_ir::AssociatedTyDatumBound { bounds, where_clauses };
    let datum = AssociatedTyDatum {
        trait_id: trait_.to_chalk(db),
        id,
        name: type_alias,
        binders: make_binders(bound_data, generic_params.len()),
    };
    Arc::new(datum)
}

pub(crate) fn trait_datum_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    trait_id: TraitId,
) -> Arc<TraitDatum> {
    debug!("trait_datum {:?}", trait_id);
    let trait_: hir_def::TraitId = from_chalk(db, trait_id);
    let trait_data = db.trait_data(trait_);
    debug!("trait {:?} = {:?}", trait_id, trait_data.name);
    let generic_params = generics(db.upcast(), trait_.into());
    let bound_vars = Substs::bound_vars(&generic_params, DebruijnIndex::INNERMOST);
    let flags = chalk_rust_ir::TraitFlags {
        auto: trait_data.auto,
        upstream: trait_.lookup(db.upcast()).container.module(db.upcast()).krate != krate,
        non_enumerable: true,
        coinductive: false, // only relevant for Chalk testing
        // FIXME set these flags correctly
        marker: false,
        fundamental: false,
    };
    let where_clauses = convert_where_clauses(db, trait_.into(), &bound_vars);
    let associated_ty_ids =
        trait_data.associated_types().map(|type_alias| type_alias.to_chalk(db)).collect();
    let trait_datum_bound = chalk_rust_ir::TraitDatumBound { where_clauses };
    let well_known = None; // FIXME set this (depending on lang items)
    let trait_datum = TraitDatum {
        id: trait_id,
        binders: make_binders(trait_datum_bound, bound_vars.len()),
        flags,
        associated_ty_ids,
        well_known,
    };
    Arc::new(trait_datum)
}

pub(crate) fn struct_datum_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    struct_id: StructId,
) -> Arc<StructDatum> {
    debug!("struct_datum {:?}", struct_id);
    let type_ctor: TypeCtor = from_chalk(db, TypeName::Adt(struct_id));
    debug!("struct {:?} = {:?}", struct_id, type_ctor);
    let num_params = type_ctor.num_ty_params(db);
    let upstream = type_ctor.krate(db) != Some(krate);
    let where_clauses = type_ctor
        .as_generic_def()
        .map(|generic_def| {
            let generic_params = generics(db.upcast(), generic_def);
            let bound_vars = Substs::bound_vars(&generic_params, DebruijnIndex::INNERMOST);
            convert_where_clauses(db, generic_def, &bound_vars)
        })
        .unwrap_or_else(Vec::new);
    let flags = chalk_rust_ir::AdtFlags {
        upstream,
        // FIXME set fundamental flag correctly
        fundamental: false,
    };
    let struct_datum_bound = chalk_rust_ir::AdtDatumBound {
        fields: Vec::new(), // FIXME add fields (only relevant for auto traits)
        where_clauses,
    };
    let struct_datum =
        StructDatum { id: struct_id, binders: make_binders(struct_datum_bound, num_params), flags };
    Arc::new(struct_datum)
}

pub(crate) fn impl_datum_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    impl_id: ImplId,
) -> Arc<ImplDatum> {
    let _p = ra_prof::profile("impl_datum");
    debug!("impl_datum {:?}", impl_id);
    let impl_: Impl = from_chalk(db, impl_id);
    match impl_ {
        Impl::ImplDef(impl_def) => impl_def_datum(db, krate, impl_id, impl_def),
        _ => Arc::new(builtin::impl_datum(db, krate, impl_).to_chalk(db)),
    }
}

fn impl_def_datum(
    db: &dyn HirDatabase,
    krate: CrateId,
    chalk_id: ImplId,
    impl_id: hir_def::ImplId,
) -> Arc<ImplDatum> {
    let trait_ref = db
        .impl_trait(impl_id)
        // ImplIds for impls where the trait ref can't be resolved should never reach Chalk
        .expect("invalid impl passed to Chalk")
        .value;
    let impl_data = db.impl_data(impl_id);

    let generic_params = generics(db.upcast(), impl_id.into());
    let bound_vars = Substs::bound_vars(&generic_params, DebruijnIndex::INNERMOST);
    let trait_ = trait_ref.trait_;
    let impl_type = if impl_id.lookup(db.upcast()).container.module(db.upcast()).krate == krate {
        chalk_rust_ir::ImplType::Local
    } else {
        chalk_rust_ir::ImplType::External
    };
    let where_clauses = convert_where_clauses(db, impl_id.into(), &bound_vars);
    let negative = impl_data.is_negative;
    debug!(
        "impl {:?}: {}{} where {:?}",
        chalk_id,
        if negative { "!" } else { "" },
        trait_ref.display(db),
        where_clauses
    );
    let trait_ref = trait_ref.to_chalk(db);

    let polarity = if negative {
        chalk_rust_ir::Polarity::Negative
    } else {
        chalk_rust_ir::Polarity::Positive
    };

    let impl_datum_bound = chalk_rust_ir::ImplDatumBound { trait_ref, where_clauses };
    let trait_data = db.trait_data(trait_);
    let associated_ty_value_ids = impl_data
        .items
        .iter()
        .filter_map(|item| match item {
            AssocItemId::TypeAliasId(type_alias) => Some(*type_alias),
            _ => None,
        })
        .filter(|&type_alias| {
            // don't include associated types that don't exist in the trait
            let name = &db.type_alias_data(type_alias).name;
            trait_data.associated_type_by_name(name).is_some()
        })
        .map(|type_alias| AssocTyValue::TypeAlias(type_alias).to_chalk(db))
        .collect();
    debug!("impl_datum: {:?}", impl_datum_bound);
    let impl_datum = ImplDatum {
        binders: make_binders(impl_datum_bound, bound_vars.len()),
        impl_type,
        polarity,
        associated_ty_value_ids,
    };
    Arc::new(impl_datum)
}

pub(crate) fn associated_ty_value_query(
    db: &dyn HirDatabase,
    krate: CrateId,
    id: AssociatedTyValueId,
) -> Arc<AssociatedTyValue> {
    let data: AssocTyValue = from_chalk(db, id);
    match data {
        AssocTyValue::TypeAlias(type_alias) => {
            type_alias_associated_ty_value(db, krate, type_alias)
        }
        _ => Arc::new(builtin::associated_ty_value(db, krate, data).to_chalk(db)),
    }
}

fn type_alias_associated_ty_value(
    db: &dyn HirDatabase,
    _krate: CrateId,
    type_alias: TypeAliasId,
) -> Arc<AssociatedTyValue> {
    let type_alias_data = db.type_alias_data(type_alias);
    let impl_id = match type_alias.lookup(db.upcast()).container {
        AssocContainerId::ImplId(it) => it,
        _ => panic!("assoc ty value should be in impl"),
    };

    let trait_ref = db.impl_trait(impl_id).expect("assoc ty value should not exist").value; // we don't return any assoc ty values if the impl'd trait can't be resolved

    let assoc_ty = db
        .trait_data(trait_ref.trait_)
        .associated_type_by_name(&type_alias_data.name)
        .expect("assoc ty value should not exist"); // validated when building the impl data as well
    let ty = db.ty(type_alias.into());
    let value_bound = chalk_rust_ir::AssociatedTyValueBound { ty: ty.value.to_chalk(db) };
    let value = chalk_rust_ir::AssociatedTyValue {
        impl_id: Impl::ImplDef(impl_id).to_chalk(db),
        associated_ty_id: assoc_ty.to_chalk(db),
        value: make_binders(value_bound, ty.num_binders),
    };
    Arc::new(value)
}

impl From<StructId> for crate::TypeCtorId {
    fn from(struct_id: StructId) -> Self {
        InternKey::from_intern_id(struct_id.0)
    }
}

impl From<crate::TypeCtorId> for StructId {
    fn from(type_ctor_id: crate::TypeCtorId) -> Self {
        chalk_ir::AdtId(type_ctor_id.as_intern_id())
    }
}

impl From<ImplId> for crate::traits::GlobalImplId {
    fn from(impl_id: ImplId) -> Self {
        InternKey::from_intern_id(impl_id.0)
    }
}

impl From<crate::traits::GlobalImplId> for ImplId {
    fn from(impl_id: crate::traits::GlobalImplId) -> Self {
        chalk_ir::ImplId(impl_id.as_intern_id())
    }
}

impl From<chalk_rust_ir::AssociatedTyValueId<Interner>> for crate::traits::AssocTyValueId {
    fn from(id: chalk_rust_ir::AssociatedTyValueId<Interner>) -> Self {
        Self::from_intern_id(id.0)
    }
}

impl From<crate::traits::AssocTyValueId> for chalk_rust_ir::AssociatedTyValueId<Interner> {
    fn from(assoc_ty_value_id: crate::traits::AssocTyValueId) -> Self {
        chalk_rust_ir::AssociatedTyValueId(assoc_ty_value_id.as_intern_id())
    }
}
