//! Implementation of the Chalk `Interner` trait, which allows customizing the
//! representation of the various objects Chalk deals with (types, goals etc.).

use super::tls;
use base_db::salsa::InternId;
use chalk_ir::{GenericArg, Goal, GoalData};
use hir_def::TypeAliasId;
use std::{fmt, sync::Arc};

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Interner;

pub type AssocTypeId = chalk_ir::AssocTypeId<Interner>;
pub type AssociatedTyDatum = chalk_solve::rust_ir::AssociatedTyDatum<Interner>;
pub type TraitId = chalk_ir::TraitId<Interner>;
pub type TraitDatum = chalk_solve::rust_ir::TraitDatum<Interner>;
pub type AdtId = chalk_ir::AdtId<Interner>;
pub type StructDatum = chalk_solve::rust_ir::AdtDatum<Interner>;
pub type ImplId = chalk_ir::ImplId<Interner>;
pub type ImplDatum = chalk_solve::rust_ir::ImplDatum<Interner>;
pub type AssociatedTyValueId = chalk_solve::rust_ir::AssociatedTyValueId<Interner>;
pub type AssociatedTyValue = chalk_solve::rust_ir::AssociatedTyValue<Interner>;
pub type FnDefId = chalk_ir::FnDefId<Interner>;
pub type FnDefDatum = chalk_solve::rust_ir::FnDefDatum<Interner>;
pub type OpaqueTyId = chalk_ir::OpaqueTyId<Interner>;
pub type OpaqueTyDatum = chalk_solve::rust_ir::OpaqueTyDatum<Interner>;

impl chalk_ir::interner::Interner for Interner {
    type InternedType = Arc<chalk_ir::TyData<Self>>;
    type InternedLifetime = chalk_ir::LifetimeData<Self>;
    type InternedConst = Arc<chalk_ir::ConstData<Self>>;
    type InternedConcreteConst = ();
    type InternedGenericArg = chalk_ir::GenericArgData<Self>;
    type InternedGoal = Arc<GoalData<Self>>;
    type InternedGoals = Vec<Goal<Self>>;
    type InternedSubstitution = Vec<GenericArg<Self>>;
    type InternedProgramClause = Arc<chalk_ir::ProgramClauseData<Self>>;
    type InternedProgramClauses = Arc<[chalk_ir::ProgramClause<Self>]>;
    type InternedQuantifiedWhereClauses = Vec<chalk_ir::QuantifiedWhereClause<Self>>;
    type InternedVariableKinds = Vec<chalk_ir::VariableKind<Self>>;
    type InternedCanonicalVarKinds = Vec<chalk_ir::CanonicalVarKind<Self>>;
    type InternedConstraints = Vec<chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>>;
    type DefId = InternId;
    type InternedAdtId = hir_def::AdtId;
    type Identifier = TypeAliasId;
    type FnAbi = ();

    fn debug_adt_id(type_kind_id: AdtId, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
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

    fn debug_fn_def_id(
        fn_def_id: chalk_ir::FnDefId<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_fn_def_id(fn_def_id, fmt)))
    }
    fn debug_const(
        constant: &chalk_ir::Const<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_const(constant, fmt)))
    }
    fn debug_variable_kinds(
        variable_kinds: &chalk_ir::VariableKinds<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_variable_kinds(variable_kinds, fmt)))
    }
    fn debug_variable_kinds_with_angles(
        variable_kinds: &chalk_ir::VariableKinds<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| {
            Some(prog?.debug_variable_kinds_with_angles(variable_kinds, fmt))
        })
    }
    fn debug_canonical_var_kinds(
        canonical_var_kinds: &chalk_ir::CanonicalVarKinds<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| {
            Some(prog?.debug_canonical_var_kinds(canonical_var_kinds, fmt))
        })
    }
    fn debug_program_clause(
        clause: &chalk_ir::ProgramClause<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_program_clause(clause, fmt)))
    }
    fn debug_program_clauses(
        clauses: &chalk_ir::ProgramClauses<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_program_clauses(clauses, fmt)))
    }
    fn debug_quantified_where_clauses(
        clauses: &chalk_ir::QuantifiedWhereClauses<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_quantified_where_clauses(clauses, fmt)))
    }

    fn intern_ty(&self, ty: chalk_ir::TyData<Self>) -> Arc<chalk_ir::TyData<Self>> {
        Arc::new(ty)
    }

    fn ty_data<'a>(&self, ty: &'a Arc<chalk_ir::TyData<Self>>) -> &'a chalk_ir::TyData<Self> {
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

    fn const_eq(&self, _ty: &Arc<chalk_ir::TyData<Self>>, _c1: &(), _c2: &()) -> bool {
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
    ) -> Arc<chalk_ir::ProgramClauseData<Self>> {
        Arc::new(data)
    }

    fn program_clause_data<'a>(
        &self,
        clause: &'a Arc<chalk_ir::ProgramClauseData<Self>>,
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

    fn intern_constraints<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>, E>>,
    ) -> Result<Self::InternedConstraints, E> {
        data.into_iter().collect()
    }

    fn constraints_data<'a>(
        &self,
        constraints: &'a Self::InternedConstraints,
    ) -> &'a [chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>] {
        constraints
    }
    fn debug_closure_id(
        _fn_def_id: chalk_ir::ClosureId<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }
    fn debug_constraints(
        _clauses: &chalk_ir::Constraints<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }
}

impl chalk_ir::interner::HasInterner for Interner {
    type Interner = Self;
}
