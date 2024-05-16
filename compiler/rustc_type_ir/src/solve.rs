use std::fmt;
use std::hash::Hash;

#[cfg(feature = "nightly")]
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::{Interner, NormalizesTo, Upcast};

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "P: Clone"),
    Copy(bound = "P: Copy"),
    Hash(bound = "P: Hash"),
    PartialEq(bound = "P: PartialEq"),
    Eq(bound = "P: Eq"),
    Debug(bound = "P: fmt::Debug")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct Goal<I: Interner, P> {
    pub param_env: I::ParamEnv,
    pub predicate: P,
}

impl<I: Interner, P> Goal<I, P> {
    pub fn new(tcx: I, param_env: I::ParamEnv, predicate: impl Upcast<I, P>) -> Goal<I, P> {
        Goal { param_env, predicate: predicate.upcast(tcx) }
    }

    /// Updates the goal to one with a different `predicate` but the same `param_env`.
    pub fn with<Q>(self, tcx: I, predicate: impl Upcast<I, Q>) -> Goal<I, Q> {
        Goal { param_env: self.param_env, predicate: predicate.upcast(tcx) }
    }
}

/// Why a specific goal has to be proven.
///
/// This is necessary as we treat nested goals different depending on
/// their source.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum GoalSource {
    Misc,
    /// We're proving a where-bound of an impl.
    ///
    /// FIXME(-Znext-solver=coinductive): Explain how and why this
    /// changes whether cycles are coinductive.
    ///
    /// This also impacts whether we erase constraints on overflow.
    /// Erasing constraints is generally very useful for perf and also
    /// results in better error messages by avoiding spurious errors.
    /// We do not erase overflow constraints in `normalizes-to` goals unless
    /// they are from an impl where-clause. This is necessary due to
    /// backwards compatability, cc trait-system-refactor-initiatitive#70.
    ImplWhereBound,
    /// Instantiating a higher-ranked goal and re-proving it.
    InstantiateHigherRanked,
}

#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = "Goal<I, P>: Clone"),
    Copy(bound = "Goal<I, P>: Copy"),
    Hash(bound = "Goal<I, P>: Hash"),
    PartialEq(bound = "Goal<I, P>: PartialEq"),
    Eq(bound = "Goal<I, P>: Eq"),
    Debug(bound = "Goal<I, P>: fmt::Debug")
)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct QueryInput<I: Interner, P> {
    pub goal: Goal<I, P>,
    pub predefined_opaques_in_body: I::PredefinedOpaques,
}
