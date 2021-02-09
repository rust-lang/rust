//! Types required for Chalk-related queries
//!
//! The primary purpose of this file is defining an implementation for the
//! `chalk_ir::interner::Interner` trait. The primary purpose of this trait, as
//! its name suggest, is to provide an abstraction boundary for creating
//! interned Chalk types.

use rustc_middle::mir::interpret::ConstValue;
use rustc_middle::ty::{self, AdtDef, TyCtxt};

use rustc_hir::def_id::DefId;
use rustc_target::spec::abi::Abi;

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone)]
pub struct RustInterner<'tcx> {
    pub tcx: TyCtxt<'tcx>,
}

/// We don't ever actually need this. It's only required for derives.
impl<'tcx> Hash for RustInterner<'tcx> {
    fn hash<H: Hasher>(&self, _state: &mut H) {}
}

/// We don't ever actually need this. It's only required for derives.
impl<'tcx> Ord for RustInterner<'tcx> {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

/// We don't ever actually need this. It's only required for derives.
impl<'tcx> PartialOrd for RustInterner<'tcx> {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        None
    }
}

/// We don't ever actually need this. It's only required for derives.
impl<'tcx> PartialEq for RustInterner<'tcx> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

/// We don't ever actually need this. It's only required for derives.
impl<'tcx> Eq for RustInterner<'tcx> {}

impl fmt::Debug for RustInterner<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RustInterner")
    }
}

// Right now, there is no interning at all. I was running into problems with
// adding interning in `ty/context.rs` for Chalk types with
// `parallel-compiler = true`. -jackh726
impl<'tcx> chalk_ir::interner::Interner for RustInterner<'tcx> {
    type InternedType = Box<chalk_ir::TyData<Self>>;
    type InternedLifetime = Box<chalk_ir::LifetimeData<Self>>;
    type InternedConst = Box<chalk_ir::ConstData<Self>>;
    type InternedConcreteConst = ConstValue<'tcx>;
    type InternedGenericArg = Box<chalk_ir::GenericArgData<Self>>;
    type InternedGoal = Box<chalk_ir::GoalData<Self>>;
    type InternedGoals = Vec<chalk_ir::Goal<Self>>;
    type InternedSubstitution = Vec<chalk_ir::GenericArg<Self>>;
    type InternedProgramClause = Box<chalk_ir::ProgramClauseData<Self>>;
    type InternedProgramClauses = Vec<chalk_ir::ProgramClause<Self>>;
    type InternedQuantifiedWhereClauses = Vec<chalk_ir::QuantifiedWhereClause<Self>>;
    type InternedVariableKinds = Vec<chalk_ir::VariableKind<Self>>;
    type InternedCanonicalVarKinds = Vec<chalk_ir::CanonicalVarKind<Self>>;
    type InternedConstraints = Vec<chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>>;
    type DefId = DefId;
    type InternedAdtId = &'tcx AdtDef;
    type Identifier = ();
    type FnAbi = Abi;

    fn debug_program_clause_implication(
        pci: &chalk_ir::ProgramClauseImplication<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        let mut write = || {
            write!(fmt, "{:?}", pci.consequence)?;

            let conditions = pci.conditions.interned();

            let conds = conditions.len();
            if conds == 0 {
                return Ok(());
            }

            write!(fmt, " :- ")?;
            for cond in &conditions[..conds - 1] {
                write!(fmt, "{:?}, ", cond)?;
            }
            write!(fmt, "{:?}", conditions[conds - 1])?;
            Ok(())
        };
        Some(write())
    }

    fn debug_substitution(
        substitution: &chalk_ir::Substitution<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", substitution.interned()))
    }

    fn debug_separator_trait_ref(
        separator_trait_ref: &chalk_ir::SeparatorTraitRef<'_, Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        let substitution = &separator_trait_ref.trait_ref.substitution;
        let parameters = substitution.interned();
        Some(write!(
            fmt,
            "{:?}{}{:?}{:?}",
            parameters[0],
            separator_trait_ref.separator,
            separator_trait_ref.trait_ref.trait_id,
            chalk_ir::debug::Angle(&parameters[1..])
        ))
    }

    fn debug_quantified_where_clauses(
        clauses: &chalk_ir::QuantifiedWhereClauses<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", clauses.interned()))
    }

    fn debug_ty(ty: &chalk_ir::Ty<Self>, fmt: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        match &ty.interned().kind {
            chalk_ir::TyKind::Ref(chalk_ir::Mutability::Not, lifetime, ty) => {
                Some(write!(fmt, "(&{:?} {:?})", lifetime, ty))
            }
            chalk_ir::TyKind::Ref(chalk_ir::Mutability::Mut, lifetime, ty) => {
                Some(write!(fmt, "(&{:?} mut {:?})", lifetime, ty))
            }
            chalk_ir::TyKind::Array(ty, len) => Some(write!(fmt, "[{:?}; {:?}]", ty, len)),
            chalk_ir::TyKind::Slice(ty) => Some(write!(fmt, "[{:?}]", ty)),
            chalk_ir::TyKind::Tuple(len, substs) => Some((|| {
                write!(fmt, "(")?;
                for (idx, substitution) in substs.interned().iter().enumerate() {
                    if idx == *len && *len != 1 {
                        // Don't add a trailing comma if the tuple has more than one element
                        write!(fmt, "{:?}", substitution)?;
                    } else {
                        write!(fmt, "{:?},", substitution)?;
                    }
                }
                write!(fmt, ")")
            })()),
            _ => None,
        }
    }

    fn debug_alias(
        alias_ty: &chalk_ir::AliasTy<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        match alias_ty {
            chalk_ir::AliasTy::Projection(projection_ty) => {
                Self::debug_projection_ty(projection_ty, fmt)
            }
            chalk_ir::AliasTy::Opaque(opaque_ty) => Self::debug_opaque_ty(opaque_ty, fmt),
        }
    }

    fn debug_projection_ty(
        projection_ty: &chalk_ir::ProjectionTy<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(
            fmt,
            "projection: {:?} {:?}",
            projection_ty.associated_ty_id, projection_ty.substitution,
        ))
    }

    fn debug_opaque_ty(
        opaque_ty: &chalk_ir::OpaqueTy<Self>,
        fmt: &mut fmt::Formatter<'_>,
    ) -> Option<fmt::Result> {
        Some(write!(fmt, "{:?}", opaque_ty.opaque_ty_id))
    }

    fn intern_ty(&self, ty: chalk_ir::TyData<Self>) -> Self::InternedType {
        Box::new(ty)
    }

    fn ty_data<'a>(&self, ty: &'a Self::InternedType) -> &'a chalk_ir::TyData<Self> {
        ty
    }

    fn intern_lifetime(&self, lifetime: chalk_ir::LifetimeData<Self>) -> Self::InternedLifetime {
        Box::new(lifetime)
    }

    fn lifetime_data<'a>(
        &self,
        lifetime: &'a Self::InternedLifetime,
    ) -> &'a chalk_ir::LifetimeData<Self> {
        &lifetime
    }

    fn intern_const(&self, constant: chalk_ir::ConstData<Self>) -> Self::InternedConst {
        Box::new(constant)
    }

    fn const_data<'a>(&self, constant: &'a Self::InternedConst) -> &'a chalk_ir::ConstData<Self> {
        &constant
    }

    fn const_eq(
        &self,
        _ty: &Self::InternedType,
        c1: &Self::InternedConcreteConst,
        c2: &Self::InternedConcreteConst,
    ) -> bool {
        c1 == c2
    }

    fn intern_generic_arg(&self, data: chalk_ir::GenericArgData<Self>) -> Self::InternedGenericArg {
        Box::new(data)
    }

    fn generic_arg_data<'a>(
        &self,
        data: &'a Self::InternedGenericArg,
    ) -> &'a chalk_ir::GenericArgData<Self> {
        &data
    }

    fn intern_goal(&self, goal: chalk_ir::GoalData<Self>) -> Self::InternedGoal {
        Box::new(goal)
    }

    fn goal_data<'a>(&self, goal: &'a Self::InternedGoal) -> &'a chalk_ir::GoalData<Self> {
        &goal
    }

    fn intern_goals<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::Goal<Self>, E>>,
    ) -> Result<Self::InternedGoals, E> {
        data.into_iter().collect::<Result<Vec<_>, _>>()
    }

    fn goals_data<'a>(&self, goals: &'a Self::InternedGoals) -> &'a [chalk_ir::Goal<Self>] {
        goals
    }

    fn intern_substitution<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::GenericArg<Self>, E>>,
    ) -> Result<Self::InternedSubstitution, E> {
        data.into_iter().collect::<Result<Vec<_>, _>>()
    }

    fn substitution_data<'a>(
        &self,
        substitution: &'a Self::InternedSubstitution,
    ) -> &'a [chalk_ir::GenericArg<Self>] {
        substitution
    }

    fn intern_program_clause(
        &self,
        data: chalk_ir::ProgramClauseData<Self>,
    ) -> Self::InternedProgramClause {
        Box::new(data)
    }

    fn program_clause_data<'a>(
        &self,
        clause: &'a Self::InternedProgramClause,
    ) -> &'a chalk_ir::ProgramClauseData<Self> {
        &clause
    }

    fn intern_program_clauses<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::ProgramClause<Self>, E>>,
    ) -> Result<Self::InternedProgramClauses, E> {
        data.into_iter().collect::<Result<Vec<_>, _>>()
    }

    fn program_clauses_data<'a>(
        &self,
        clauses: &'a Self::InternedProgramClauses,
    ) -> &'a [chalk_ir::ProgramClause<Self>] {
        clauses
    }

    fn intern_quantified_where_clauses<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::QuantifiedWhereClause<Self>, E>>,
    ) -> Result<Self::InternedQuantifiedWhereClauses, E> {
        data.into_iter().collect::<Result<Vec<_>, _>>()
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
        data.into_iter().collect::<Result<Vec<_>, _>>()
    }

    fn variable_kinds_data<'a>(
        &self,
        parameter_kinds: &'a Self::InternedVariableKinds,
    ) -> &'a [chalk_ir::VariableKind<Self>] {
        parameter_kinds
    }

    fn intern_canonical_var_kinds<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::CanonicalVarKind<Self>, E>>,
    ) -> Result<Self::InternedCanonicalVarKinds, E> {
        data.into_iter().collect::<Result<Vec<_>, _>>()
    }

    fn canonical_var_kinds_data<'a>(
        &self,
        canonical_var_kinds: &'a Self::InternedCanonicalVarKinds,
    ) -> &'a [chalk_ir::CanonicalVarKind<Self>] {
        canonical_var_kinds
    }

    fn intern_constraints<E>(
        &self,
        data: impl IntoIterator<Item = Result<chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>, E>>,
    ) -> Result<Self::InternedConstraints, E> {
        data.into_iter().collect::<Result<Vec<_>, _>>()
    }

    fn constraints_data<'a>(
        &self,
        constraints: &'a Self::InternedConstraints,
    ) -> &'a [chalk_ir::InEnvironment<chalk_ir::Constraint<Self>>] {
        constraints
    }
}

impl<'tcx> chalk_ir::interner::HasInterner for RustInterner<'tcx> {
    type Interner = Self;
}

/// A chalk environment and goal.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable, TypeFoldable)]
pub struct ChalkEnvironmentAndGoal<'tcx> {
    pub environment: &'tcx ty::List<ty::Predicate<'tcx>>,
    pub goal: ty::Predicate<'tcx>,
}

impl<'tcx> fmt::Display for ChalkEnvironmentAndGoal<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "environment: {:?}, goal: {}", self.environment, self.goal)
    }
}
