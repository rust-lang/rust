//! Implementation of the Chalk `Interner` trait, which allows customizing the
//! representation of the various objects Chalk deals with (types, goals etc.).

use crate::{chalk_db, tls, ConstScalar, GenericArg};
use base_db::salsa::InternId;
use chalk_ir::{Goal, GoalData};
use hir_def::TypeAliasId;
use intern::{impl_internable, Interned};
use smallvec::SmallVec;
use std::fmt;
use triomphe::Arc;

#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Interner;

#[derive(PartialEq, Eq, Hash)]
pub struct InternedWrapper<T>(T);

impl<T: fmt::Debug> fmt::Debug for InternedWrapper<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<T> std::ops::Deref for InternedWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl_internable!(
    InternedWrapper<Vec<chalk_ir::VariableKind<Interner>>>,
    InternedWrapper<SmallVec<[GenericArg; 2]>>,
    InternedWrapper<chalk_ir::TyData<Interner>>,
    InternedWrapper<chalk_ir::LifetimeData<Interner>>,
    InternedWrapper<chalk_ir::ConstData<Interner>>,
    InternedWrapper<ConstScalar>,
    InternedWrapper<Vec<chalk_ir::CanonicalVarKind<Interner>>>,
    InternedWrapper<Vec<chalk_ir::ProgramClause<Interner>>>,
    InternedWrapper<Vec<chalk_ir::QuantifiedWhereClause<Interner>>>,
    InternedWrapper<Vec<chalk_ir::Variance>>,
);

impl chalk_ir::interner::Interner for Interner {
    type InternedType = Interned<InternedWrapper<chalk_ir::TyData<Self>>>;
    type InternedLifetime = Interned<InternedWrapper<chalk_ir::LifetimeData<Self>>>;
    type InternedConst = Interned<InternedWrapper<chalk_ir::ConstData<Self>>>;
    type InternedConcreteConst = ConstScalar;
    type InternedGenericArg = chalk_ir::GenericArgData<Self>;
    type InternedGoal = Arc<GoalData<Self>>;
    type InternedGoals = Vec<Goal<Self>>;
    type InternedSubstitution = Interned<InternedWrapper<SmallVec<[GenericArg; 2]>>>;
    type InternedProgramClauses = Interned<InternedWrapper<Vec<chalk_ir::ProgramClause<Self>>>>;
    type InternedProgramClause = chalk_ir::ProgramClauseData<Self>;
    type InternedQuantifiedWhereClauses =
        Interned<InternedWrapper<Vec<chalk_ir::QuantifiedWhereClause<Self>>>>;
    type InternedVariableKinds = Interned<InternedWrapper<Vec<chalk_ir::VariableKind<Interner>>>>;
    type InternedCanonicalVarKinds =
        Interned<InternedWrapper<Vec<chalk_ir::CanonicalVarKind<Self>>>>;
    type InternedConstraints = Vec<chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>>;
    type InternedVariances = Interned<InternedWrapper<Vec<chalk_ir::Variance>>>;
    type DefId = InternId;
    type InternedAdtId = hir_def::AdtId;
    type Identifier = TypeAliasId;
    type FnAbi = ();

    fn debug_adt_id(
        type_kind_id: chalk_db::AdtId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_struct_id(type_kind_id, fmt)))
    }

    fn debug_trait_id(
        type_kind_id: chalk_db::TraitId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_trait_id(type_kind_id, fmt)))
    }

    fn debug_assoc_type_id(
        id: chalk_db::AssocTypeId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_assoc_type_id(id, fmt)))
    }

    fn debug_opaque_ty_id(
        opaque_ty_id: chalk_ir::OpaqueTyId<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "OpaqueTy#{}", opaque_ty_id.0))
    }

    fn debug_fn_def_id(
        fn_def_id: chalk_ir::FnDefId<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_fn_def_id(fn_def_id, fmt)))
    }

    fn debug_closure_id(
        _fn_def_id: chalk_ir::ClosureId<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn debug_alias(
        alias: &chalk_ir::AliasTy<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        use std::fmt::Debug;
        match alias {
            chalk_ir::AliasTy::Projection(projection_ty) => {
                Interner::debug_projection_ty(projection_ty, fmt)
            }
            chalk_ir::AliasTy::Opaque(opaque_ty) => Some(opaque_ty.fmt(fmt)),
        }
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
        Some(write!(fmt, "{:?}", opaque_ty.opaque_ty_id))
    }

    fn debug_ty(ty: &chalk_ir::Ty<Interner>, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", ty.data(Interner)))
    }

    fn debug_lifetime(
        lifetime: &chalk_ir::Lifetime<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", lifetime.data(Interner)))
    }

    fn debug_const(
        constant: &chalk_ir::Const<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", constant.data(Interner)))
    }

    fn debug_generic_arg(
        parameter: &GenericArg,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", parameter.data(Interner).inner_debug()))
    }

    fn debug_variable_kinds(
        variable_kinds: &chalk_ir::VariableKinds<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", variable_kinds.as_slice(Interner)))
    }

    fn debug_variable_kinds_with_angles(
        variable_kinds: &chalk_ir::VariableKinds<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", variable_kinds.inner_debug(Interner)))
    }

    fn debug_canonical_var_kinds(
        canonical_var_kinds: &chalk_ir::CanonicalVarKinds<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", canonical_var_kinds.as_slice(Interner)))
    }
    fn debug_goal(goal: &Goal<Interner>, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        let goal_data = goal.data(Interner);
        Some(write!(fmt, "{goal_data:?}"))
    }
    fn debug_goals(
        goals: &chalk_ir::Goals<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", goals.debug(Interner)))
    }
    fn debug_program_clause_implication(
        pci: &chalk_ir::ProgramClauseImplication<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", pci.debug(Interner)))
    }
    fn debug_program_clause(
        clause: &chalk_ir::ProgramClause<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clause.data(Interner)))
    }
    fn debug_program_clauses(
        clauses: &chalk_ir::ProgramClauses<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.as_slice(Interner)))
    }
    fn debug_substitution(
        substitution: &chalk_ir::Substitution<Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", substitution.debug(Interner)))
    }
    fn debug_separator_trait_ref(
        separator_trait_ref: &chalk_ir::SeparatorTraitRef<'_, Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", separator_trait_ref.debug(Interner)))
    }

    fn debug_quantified_where_clauses(
        clauses: &chalk_ir::QuantifiedWhereClauses<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.as_slice(Interner)))
    }

    fn debug_constraints(
        _clauses: &chalk_ir::Constraints<Self>,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn intern_ty(self, kind: chalk_ir::TyKind<Self>) -> Self::InternedType {
        let flags = kind.compute_flags(self);
        Interned::new(InternedWrapper(chalk_ir::TyData { kind, flags }))
    }

    fn ty_data(self, ty: &Self::InternedType) -> &chalk_ir::TyData<Self> {
        &ty.0
    }

    fn intern_lifetime(self, lifetime: chalk_ir::LifetimeData<Self>) -> Self::InternedLifetime {
        Interned::new(InternedWrapper(lifetime))
    }

    fn lifetime_data(self, lifetime: &Self::InternedLifetime) -> &chalk_ir::LifetimeData<Self> {
        &lifetime.0
    }

    fn intern_const(self, constant: chalk_ir::ConstData<Self>) -> Self::InternedConst {
        Interned::new(InternedWrapper(constant))
    }

    fn const_data(self, constant: &Self::InternedConst) -> &chalk_ir::ConstData<Self> {
        &constant.0
    }

    fn const_eq(
        self,
        _ty: &Self::InternedType,
        c1: &Self::InternedConcreteConst,
        c2: &Self::InternedConcreteConst,
    ) -> bool {
        !matches!(c1, ConstScalar::Bytes(..)) || !matches!(c2, ConstScalar::Bytes(..)) || (c1 == c2)
    }

    fn intern_generic_arg(
        self,
        parameter: chalk_ir::GenericArgData<Self>,
    ) -> Self::InternedGenericArg {
        parameter
    }

    fn generic_arg_data(
        self,
        parameter: &Self::InternedGenericArg,
    ) -> &chalk_ir::GenericArgData<Self> {
        parameter
    }

    fn intern_goal(self, goal: GoalData<Self>) -> Self::InternedGoal {
        Arc::new(goal)
    }

    fn goal_data(self, goal: &Self::InternedGoal) -> &GoalData<Self> {
        goal
    }

    fn intern_goals<E>(
        self,
        data: impl IntoIterator<Item = Result<Goal<Self>, E>>,
    ) -> Result<Self::InternedGoals, E> {
        data.into_iter().collect()
    }

    fn goals_data(self, goals: &Self::InternedGoals) -> &[Goal<Interner>] {
        goals
    }

    fn intern_substitution<E>(
        self,
        data: impl IntoIterator<Item = Result<GenericArg, E>>,
    ) -> Result<Self::InternedSubstitution, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn substitution_data(self, substitution: &Self::InternedSubstitution) -> &[GenericArg] {
        &substitution.as_ref().0
    }

    fn intern_program_clause(
        self,
        data: chalk_ir::ProgramClauseData<Self>,
    ) -> Self::InternedProgramClause {
        data
    }

    fn program_clause_data(
        self,
        clause: &Self::InternedProgramClause,
    ) -> &chalk_ir::ProgramClauseData<Self> {
        clause
    }

    fn intern_program_clauses<E>(
        self,
        data: impl IntoIterator<Item = Result<chalk_ir::ProgramClause<Self>, E>>,
    ) -> Result<Self::InternedProgramClauses, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn program_clauses_data(
        self,
        clauses: &Self::InternedProgramClauses,
    ) -> &[chalk_ir::ProgramClause<Self>] {
        clauses
    }

    fn intern_quantified_where_clauses<E>(
        self,
        data: impl IntoIterator<Item = Result<chalk_ir::QuantifiedWhereClause<Self>, E>>,
    ) -> Result<Self::InternedQuantifiedWhereClauses, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn quantified_where_clauses_data(
        self,
        clauses: &Self::InternedQuantifiedWhereClauses,
    ) -> &[chalk_ir::QuantifiedWhereClause<Self>] {
        clauses
    }

    fn intern_generic_arg_kinds<E>(
        self,
        data: impl IntoIterator<Item = Result<chalk_ir::VariableKind<Self>, E>>,
    ) -> Result<Self::InternedVariableKinds, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn variable_kinds_data(
        self,
        parameter_kinds: &Self::InternedVariableKinds,
    ) -> &[chalk_ir::VariableKind<Self>] {
        &parameter_kinds.as_ref().0
    }

    fn intern_canonical_var_kinds<E>(
        self,
        data: impl IntoIterator<Item = Result<chalk_ir::CanonicalVarKind<Self>, E>>,
    ) -> Result<Self::InternedCanonicalVarKinds, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn canonical_var_kinds_data(
        self,
        canonical_var_kinds: &Self::InternedCanonicalVarKinds,
    ) -> &[chalk_ir::CanonicalVarKind<Self>] {
        canonical_var_kinds
    }
    fn intern_constraints<E>(
        self,
        data: impl IntoIterator<Item = Result<chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>, E>>,
    ) -> Result<Self::InternedConstraints, E> {
        data.into_iter().collect()
    }
    fn constraints_data(
        self,
        constraints: &Self::InternedConstraints,
    ) -> &[chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>] {
        constraints
    }

    fn intern_variances<E>(
        self,
        data: impl IntoIterator<Item = Result<chalk_ir::Variance, E>>,
    ) -> Result<Self::InternedVariances, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn variances_data(self, variances: &Self::InternedVariances) -> &[chalk_ir::Variance] {
        variances
    }
}

impl chalk_ir::interner::HasInterner for Interner {
    type Interner = Self;
}

#[macro_export]
macro_rules! has_interner {
    ($t:ty) => {
        impl HasInterner for $t {
            type Interner = crate::Interner;
        }
    };
}
