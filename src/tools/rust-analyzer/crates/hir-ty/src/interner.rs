//! Implementation of the Chalk `Interner` trait, which allows customizing the
//! representation of the various objects Chalk deals with (types, goals etc.).

use crate::{
    AliasTy, CanonicalVarKind, CanonicalVarKinds, ClosureId, Const, ConstData, ConstScalar,
    Constraint, Constraints, FnAbi, FnDefId, GenericArg, GenericArgData, Goal, GoalData, Goals,
    InEnvironment, Lifetime, LifetimeData, OpaqueTy, OpaqueTyId, ProgramClause, ProgramClauseData,
    ProgramClauses, ProjectionTy, QuantifiedWhereClause, QuantifiedWhereClauses, Substitution, Ty,
    TyData, TyKind, VariableKind, VariableKinds, chalk_db, tls,
};
use chalk_ir::{ProgramClauseImplication, SeparatorTraitRef, Variance};
use hir_def::TypeAliasId;
use intern::{Interned, impl_internable};
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
    InternedWrapper<Vec<VariableKind>>,
    InternedWrapper<SmallVec<[GenericArg; 2]>>,
    InternedWrapper<TyData>,
    InternedWrapper<LifetimeData>,
    InternedWrapper<ConstData>,
    InternedWrapper<ConstScalar>,
    InternedWrapper<Vec<CanonicalVarKind>>,
    InternedWrapper<Box<[ProgramClause]>>,
    InternedWrapper<Vec<QuantifiedWhereClause>>,
    InternedWrapper<SmallVec<[Variance; 16]>>,
);

impl chalk_ir::interner::Interner for Interner {
    type InternedType = Interned<InternedWrapper<TyData>>;
    type InternedLifetime = Interned<InternedWrapper<LifetimeData>>;
    type InternedConst = Interned<InternedWrapper<ConstData>>;
    type InternedConcreteConst = ConstScalar;
    type InternedGenericArg = GenericArgData;
    // We could do the following, but that saves "only" 20mb on self while increasing inference
    // time by ~2.5%
    // type InternedGoal = Interned<InternedWrapper<GoalData>>;
    type InternedGoal = Arc<GoalData>;
    type InternedGoals = Vec<Goal>;
    type InternedSubstitution = Interned<InternedWrapper<SmallVec<[GenericArg; 2]>>>;
    type InternedProgramClauses = Interned<InternedWrapper<Box<[ProgramClause]>>>;
    type InternedProgramClause = ProgramClauseData;
    type InternedQuantifiedWhereClauses = Interned<InternedWrapper<Vec<QuantifiedWhereClause>>>;
    type InternedVariableKinds = Interned<InternedWrapper<Vec<VariableKind>>>;
    type InternedCanonicalVarKinds = Interned<InternedWrapper<Vec<CanonicalVarKind>>>;
    type InternedConstraints = Vec<InEnvironment<Constraint>>;
    type InternedVariances = SmallVec<[Variance; 16]>;
    type DefId = salsa::Id;
    type InternedAdtId = hir_def::AdtId;
    type Identifier = TypeAliasId;
    type FnAbi = FnAbi;

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
        opaque_ty_id: OpaqueTyId,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "OpaqueTy#{:?}", opaque_ty_id.0))
    }

    fn debug_fn_def_id(fn_def_id: FnDefId, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_fn_def_id(fn_def_id, fmt)))
    }

    fn debug_closure_id(
        _fn_def_id: ClosureId,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn debug_alias(alias: &AliasTy, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        use std::fmt::Debug;
        match alias {
            AliasTy::Projection(projection_ty) => Interner::debug_projection_ty(projection_ty, fmt),
            AliasTy::Opaque(opaque_ty) => Some(opaque_ty.fmt(fmt)),
        }
    }

    fn debug_projection_ty(
        proj: &ProjectionTy,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        tls::with_current_program(|prog| Some(prog?.debug_projection_ty(proj, fmt)))
    }

    fn debug_opaque_ty(opaque_ty: &OpaqueTy, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", opaque_ty.opaque_ty_id))
    }

    fn debug_ty(ty: &Ty, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", ty.data(Interner)))
    }

    fn debug_lifetime(lifetime: &Lifetime, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", lifetime.data(Interner)))
    }

    fn debug_const(constant: &Const, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", constant.data(Interner)))
    }

    fn debug_generic_arg(
        parameter: &GenericArg,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", parameter.data(Interner).inner_debug()))
    }

    fn debug_variable_kinds(
        variable_kinds: &VariableKinds,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", variable_kinds.as_slice(Interner)))
    }

    fn debug_variable_kinds_with_angles(
        variable_kinds: &VariableKinds,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", variable_kinds.inner_debug(Interner)))
    }

    fn debug_canonical_var_kinds(
        canonical_var_kinds: &CanonicalVarKinds,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", canonical_var_kinds.as_slice(Interner)))
    }
    fn debug_goal(goal: &Goal, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        let goal_data = goal.data(Interner);
        Some(write!(fmt, "{goal_data:?}"))
    }
    fn debug_goals(goals: &Goals, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", goals.debug(Interner)))
    }
    fn debug_program_clause_implication(
        pci: &ProgramClauseImplication<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", pci.debug(Interner)))
    }
    fn debug_program_clause(
        clause: &ProgramClause,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clause.data(Interner)))
    }
    fn debug_program_clauses(
        clauses: &ProgramClauses,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.as_slice(Interner)))
    }
    fn debug_substitution(
        substitution: &Substitution,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", substitution.debug(Interner)))
    }
    fn debug_separator_trait_ref(
        separator_trait_ref: &SeparatorTraitRef<'_, Interner>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", separator_trait_ref.debug(Interner)))
    }

    fn debug_quantified_where_clauses(
        clauses: &QuantifiedWhereClauses,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.as_slice(Interner)))
    }

    fn debug_constraints(
        _clauses: &Constraints,
        _fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        None
    }

    fn intern_ty(self, kind: TyKind) -> Self::InternedType {
        let flags = kind.compute_flags(self);
        Interned::new(InternedWrapper(TyData { kind, flags }))
    }

    fn ty_data(self, ty: &Self::InternedType) -> &TyData {
        &ty.0
    }

    fn intern_lifetime(self, lifetime: LifetimeData) -> Self::InternedLifetime {
        Interned::new(InternedWrapper(lifetime))
    }

    fn lifetime_data(self, lifetime: &Self::InternedLifetime) -> &LifetimeData {
        &lifetime.0
    }

    fn intern_const(self, constant: ConstData) -> Self::InternedConst {
        Interned::new(InternedWrapper(constant))
    }

    fn const_data(self, constant: &Self::InternedConst) -> &ConstData {
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

    fn intern_generic_arg(self, parameter: GenericArgData) -> Self::InternedGenericArg {
        parameter
    }

    fn generic_arg_data(self, parameter: &Self::InternedGenericArg) -> &GenericArgData {
        parameter
    }

    fn intern_goal(self, goal: GoalData) -> Self::InternedGoal {
        Arc::new(goal)
    }

    fn goal_data(self, goal: &Self::InternedGoal) -> &GoalData {
        goal
    }

    fn intern_goals<E>(
        self,
        data: impl IntoIterator<Item = Result<Goal, E>>,
    ) -> Result<Self::InternedGoals, E> {
        // let hash =
        //     std::hash::BuildHasher::hash_one(&BuildHasherDefault::<FxHasher>::default(), &goal);
        // Interned::new(InternedWrapper(PreHashedWrapper(goal, hash)))
        data.into_iter().collect()
    }

    fn goals_data(self, goals: &Self::InternedGoals) -> &[Goal] {
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

    fn intern_program_clause(self, data: ProgramClauseData) -> Self::InternedProgramClause {
        data
    }

    fn program_clause_data(self, clause: &Self::InternedProgramClause) -> &ProgramClauseData {
        clause
    }

    fn intern_program_clauses<E>(
        self,
        data: impl IntoIterator<Item = Result<ProgramClause, E>>,
    ) -> Result<Self::InternedProgramClauses, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn program_clauses_data(self, clauses: &Self::InternedProgramClauses) -> &[ProgramClause] {
        clauses
    }

    fn intern_quantified_where_clauses<E>(
        self,
        data: impl IntoIterator<Item = Result<QuantifiedWhereClause, E>>,
    ) -> Result<Self::InternedQuantifiedWhereClauses, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn quantified_where_clauses_data(
        self,
        clauses: &Self::InternedQuantifiedWhereClauses,
    ) -> &[QuantifiedWhereClause] {
        clauses
    }

    fn intern_generic_arg_kinds<E>(
        self,
        data: impl IntoIterator<Item = Result<VariableKind, E>>,
    ) -> Result<Self::InternedVariableKinds, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn variable_kinds_data(self, parameter_kinds: &Self::InternedVariableKinds) -> &[VariableKind] {
        &parameter_kinds.as_ref().0
    }

    fn intern_canonical_var_kinds<E>(
        self,
        data: impl IntoIterator<Item = Result<CanonicalVarKind, E>>,
    ) -> Result<Self::InternedCanonicalVarKinds, E> {
        Ok(Interned::new(InternedWrapper(data.into_iter().collect::<Result<_, _>>()?)))
    }

    fn canonical_var_kinds_data(
        self,
        canonical_var_kinds: &Self::InternedCanonicalVarKinds,
    ) -> &[CanonicalVarKind] {
        canonical_var_kinds
    }
    fn intern_constraints<E>(
        self,
        data: impl IntoIterator<Item = Result<InEnvironment<Constraint>, E>>,
    ) -> Result<Self::InternedConstraints, E> {
        data.into_iter().collect()
    }
    fn constraints_data(
        self,
        constraints: &Self::InternedConstraints,
    ) -> &[InEnvironment<Constraint>] {
        constraints
    }

    fn intern_variances<E>(
        self,
        data: impl IntoIterator<Item = Result<Variance, E>>,
    ) -> Result<Self::InternedVariances, E> {
        data.into_iter().collect::<Result<_, _>>()
    }

    fn variances_data(self, variances: &Self::InternedVariances) -> &[Variance] {
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
            type Interner = $crate::Interner;
        }
    };
}
