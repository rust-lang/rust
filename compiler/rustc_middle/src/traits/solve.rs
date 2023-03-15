use std::ops::ControlFlow;

use rustc_data_structures::intern::Interned;
use rustc_query_system::cache::Cache;

use crate::infer::canonical::{CanonicalVarValues, QueryRegionConstraints};
use crate::traits::query::NoSolution;
use crate::traits::Canonical;
use crate::ty::{
    self, FallibleTypeFolder, ToPredicate, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeVisitable,
    TypeVisitor,
};

pub type EvaluationCache<'tcx> = Cache<CanonicalGoal<'tcx>, QueryResult<'tcx>>;

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub struct Goal<'tcx, P> {
    pub param_env: ty::ParamEnv<'tcx>,
    pub predicate: P,
}

impl<'tcx, P> Goal<'tcx, P> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: impl ToPredicate<'tcx, P>,
    ) -> Goal<'tcx, P> {
        Goal { param_env, predicate: predicate.to_predicate(tcx) }
    }

    /// Updates the goal to one with a different `predicate` but the same `param_env`.
    pub fn with<Q>(self, tcx: TyCtxt<'tcx>, predicate: impl ToPredicate<'tcx, Q>) -> Goal<'tcx, Q> {
        Goal { param_env: self.param_env, predicate: predicate.to_predicate(tcx) }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub struct Response<'tcx> {
    pub var_values: CanonicalVarValues<'tcx>,
    /// Additional constraints returned by this query.
    pub external_constraints: ExternalConstraints<'tcx>,
    pub certainty: Certainty,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub enum Certainty {
    Yes,
    Maybe(MaybeCause),
}

impl Certainty {
    pub const AMBIGUOUS: Certainty = Certainty::Maybe(MaybeCause::Ambiguity);

    /// When proving multiple goals using **AND**, e.g. nested obligations for an impl,
    /// use this function to unify the certainty of these goals
    pub fn unify_and(self, other: Certainty) -> Certainty {
        match (self, other) {
            (Certainty::Yes, Certainty::Yes) => Certainty::Yes,
            (Certainty::Yes, Certainty::Maybe(_)) => other,
            (Certainty::Maybe(_), Certainty::Yes) => self,
            (Certainty::Maybe(MaybeCause::Overflow), Certainty::Maybe(MaybeCause::Overflow)) => {
                Certainty::Maybe(MaybeCause::Overflow)
            }
            // If at least one of the goals is ambiguous, hide the overflow as the ambiguous goal
            // may still result in failure.
            (Certainty::Maybe(MaybeCause::Ambiguity), Certainty::Maybe(_))
            | (Certainty::Maybe(_), Certainty::Maybe(MaybeCause::Ambiguity)) => {
                Certainty::Maybe(MaybeCause::Ambiguity)
            }
        }
    }
}

/// Why we failed to evaluate a goal.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, TypeFoldable, TypeVisitable)]
pub enum MaybeCause {
    /// We failed due to ambiguity. This ambiguity can either
    /// be a true ambiguity, i.e. there are multiple different answers,
    /// or we hit a case where we just don't bother, e.g. `?x: Trait` goals.
    Ambiguity,
    /// We gave up due to an overflow, most often by hitting the recursion limit.
    Overflow,
}

pub type CanonicalGoal<'tcx, T = ty::Predicate<'tcx>> = Canonical<'tcx, Goal<'tcx, T>>;

pub type CanonicalResponse<'tcx> = Canonical<'tcx, Response<'tcx>>;

/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<'tcx> = Result<CanonicalResponse<'tcx>, NoSolution>;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub struct ExternalConstraints<'tcx>(pub(crate) Interned<'tcx, ExternalConstraintsData<'tcx>>);

impl<'tcx> std::ops::Deref for ExternalConstraints<'tcx> {
    type Target = ExternalConstraintsData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, Default, TypeFoldable, TypeVisitable)]
pub struct ExternalConstraintsData<'tcx> {
    // FIXME: implement this.
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub opaque_types: Vec<(Ty<'tcx>, Ty<'tcx>)>,
}

// FIXME: Having to clone `region_constraints` for folding feels bad and
// probably isn't great wrt performance.
//
// Not sure how to fix this, maybe we should also intern `opaque_types` and
// `region_constraints` here or something.
impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().try_fold_with(folder)?,
            opaque_types: self
                .opaque_types
                .iter()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
        }))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        TypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().fold_with(folder),
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(
        &self,
        visitor: &mut V,
    ) -> std::ops::ControlFlow<V::BreakTy> {
        self.region_constraints.visit_with(visitor)?;
        self.opaque_types.visit_with(visitor)?;
        ControlFlow::Continue(())
    }
}
