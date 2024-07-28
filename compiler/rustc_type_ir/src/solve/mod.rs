pub mod inspect;

use std::fmt;
use std::hash::Hash;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::{self as ty, Canonical, CanonicalVarValues, Interner, Upcast};

/// Depending on the stage of compilation, we want projection to be
/// more or less conservative.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub enum Reveal {
    /// At type-checking time, we refuse to project any associated
    /// type that is marked `default`. Non-`default` ("final") types
    /// are always projected. This is necessary in general for
    /// soundness of specialization. However, we *could* allow
    /// projections in fully-monomorphic cases. We choose not to,
    /// because we prefer for `default type` to force the type
    /// definition to be treated abstractly by any consumers of the
    /// impl. Concretely, that means that the following example will
    /// fail to compile:
    ///
    /// ```compile_fail,E0308
    /// #![feature(specialization)]
    /// trait Assoc {
    ///     type Output;
    /// }
    ///
    /// impl<T> Assoc for T {
    ///     default type Output = bool;
    /// }
    ///
    /// fn main() {
    ///     let x: <() as Assoc>::Output = true;
    /// }
    /// ```
    ///
    /// We also do not reveal the hidden type of opaque types during
    /// type-checking.
    UserFacing,

    /// At codegen time, all monomorphic projections will succeed.
    /// Also, `impl Trait` is normalized to the concrete type,
    /// which has to be already collected by type-checking.
    ///
    /// NOTE: as `impl Trait`'s concrete type should *never*
    /// be observable directly by the user, `Reveal::All`
    /// should not be used by checks which may expose
    /// type equality or type contents to the user.
    /// There are some exceptions, e.g., around auto traits and
    /// transmute-checking, which expose some details, but
    /// not the whole concrete type of the `impl Trait`.
    All,
}

#[derive(Debug, Clone, Copy)]
pub enum SolverMode {
    /// Ordinary trait solving, using everywhere except for coherence.
    Normal,
    /// Trait solving during coherence. There are a few notable differences
    /// between coherence and ordinary trait solving.
    ///
    /// Most importantly, trait solving during coherence must not be incomplete,
    /// i.e. return `Err(NoSolution)` for goals for which a solution exists.
    /// This means that we must not make any guesses or arbitrary choices.
    Coherence,
}

pub type CanonicalInput<I, T = <I as Interner>::Predicate> = Canonical<I, QueryInput<I, T>>;
pub type CanonicalResponse<I> = Canonical<I, Response<I>>;
/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<I> = Result<CanonicalResponse<I>, NoSolution>;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct NoSolution;

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive_where(Clone; I: Interner, P: Clone)]
#[derive_where(Copy; I: Interner, P: Copy)]
#[derive_where(Hash; I: Interner, P: Hash)]
#[derive_where(PartialEq; I: Interner, P: PartialEq)]
#[derive_where(Eq; I: Interner, P: Eq)]
#[derive_where(Debug; I: Interner, P: fmt::Debug)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct Goal<I: Interner, P> {
    pub param_env: I::ParamEnv,
    pub predicate: P,
}

impl<I: Interner, P> Goal<I, P> {
    pub fn new(cx: I, param_env: I::ParamEnv, predicate: impl Upcast<I, P>) -> Goal<I, P> {
        Goal { param_env, predicate: predicate.upcast(cx) }
    }

    /// Updates the goal to one with a different `predicate` but the same `param_env`.
    pub fn with<Q>(self, cx: I, predicate: impl Upcast<I, Q>) -> Goal<I, Q> {
        Goal { param_env: self.param_env, predicate: predicate.upcast(cx) }
    }
}

/// Why a specific goal has to be proven.
///
/// This is necessary as we treat nested goals different depending on
/// their source.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

#[derive_where(Clone; I: Interner, Goal<I, P>: Clone)]
#[derive_where(Copy; I: Interner, Goal<I, P>: Copy)]
#[derive_where(Hash; I: Interner, Goal<I, P>: Hash)]
#[derive_where(PartialEq; I: Interner, Goal<I, P>: PartialEq)]
#[derive_where(Eq; I: Interner, Goal<I, P>: Eq)]
#[derive_where(Debug; I: Interner, Goal<I, P>: fmt::Debug)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct QueryInput<I: Interner, P> {
    pub goal: Goal<I, P>,
    pub predefined_opaques_in_body: I::PredefinedOpaques,
}

/// Opaques that are defined in the inference context before a query is called.
#[derive_where(Clone, Hash, PartialEq, Eq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct PredefinedOpaquesData<I: Interner> {
    pub opaque_types: Vec<(ty::OpaqueTypeKey<I>, I::Ty)>,
}

/// Possible ways the given goal can be proven.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
pub enum CandidateSource<I: Interner> {
    /// A user written impl.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// fn main() {
    ///     let x: Vec<u32> = Vec::new();
    ///     // This uses the impl from the standard library to prove `Vec<T>: Clone`.
    ///     let y = x.clone();
    /// }
    /// ```
    Impl(I::DefId),
    /// A builtin impl generated by the compiler. When adding a new special
    /// trait, try to use actual impls whenever possible. Builtin impls should
    /// only be used in cases where the impl cannot be manually be written.
    ///
    /// Notable examples are auto traits, `Sized`, and `DiscriminantKind`.
    /// For a list of all traits with builtin impls, check out the
    /// `EvalCtxt::assemble_builtin_impl_candidates` method.
    BuiltinImpl(BuiltinImplSource),
    /// An assumption from the environment.
    ///
    /// More precisely we've used the `n-th` assumption in the `param_env`.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// fn is_clone<T: Clone>(x: T) -> (T, T) {
    ///     // This uses the assumption `T: Clone` from the `where`-bounds
    ///     // to prove `T: Clone`.
    ///     (x.clone(), x)
    /// }
    /// ```
    ParamEnv(usize),
    /// If the self type is an alias type, e.g. an opaque type or a projection,
    /// we know the bounds on that alias to hold even without knowing its concrete
    /// underlying type.
    ///
    /// More precisely this candidate is using the `n-th` bound in the `item_bounds` of
    /// the self type.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// trait Trait {
    ///     type Assoc: Clone;
    /// }
    ///
    /// fn foo<T: Trait>(x: <T as Trait>::Assoc) {
    ///     // We prove `<T as Trait>::Assoc` by looking at the bounds on `Assoc` in
    ///     // in the trait definition.
    ///     let _y = x.clone();
    /// }
    /// ```
    AliasBound,
    /// A candidate that is registered only during coherence to represent some
    /// yet-unknown impl that could be produced downstream without violating orphan
    /// rules.
    // FIXME: Merge this with the forced ambiguity candidates, so those don't use `Misc`.
    CoherenceUnknowable,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext, TyEncodable, TyDecodable))]
pub enum BuiltinImplSource {
    /// Some built-in impl we don't need to differentiate. This should be used
    /// unless more specific information is necessary.
    Misc,
    /// A built-in impl for trait objects. The index is only used in winnowing.
    Object(usize),
    /// A built-in implementation of `Upcast` for trait objects to other trait objects.
    ///
    /// This can be removed when `feature(dyn_upcasting)` is stabilized, since we only
    /// use it to detect when upcasting traits in hir typeck.
    TraitUpcasting,
    /// Unsizing a tuple like `(A, B, ..., X)` to `(A, B, ..., Y)` if `X` unsizes to `Y`.
    ///
    /// This can be removed when `feature(tuple_unsizing)` is stabilized, since we only
    /// use it to detect when unsizing tuples in hir typeck.
    TupleUnsizing,
}

#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct Response<I: Interner> {
    pub certainty: Certainty,
    pub var_values: CanonicalVarValues<I>,
    /// Additional constraints returned by this query.
    pub external_constraints: I::ExternalConstraints,
}

/// Additional constraints returned on success.
#[derive_where(Clone, Hash, PartialEq, Eq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct ExternalConstraintsData<I: Interner> {
    pub region_constraints: Vec<ty::OutlivesPredicate<I, I::GenericArg>>,
    pub opaque_types: Vec<(ty::OpaqueTypeKey<I>, I::Ty)>,
    pub normalization_nested_goals: NestedNormalizationGoals<I>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct NestedNormalizationGoals<I: Interner>(pub Vec<(GoalSource, Goal<I, I::Predicate>)>);

impl<I: Interner> NestedNormalizationGoals<I> {
    pub fn empty() -> Self {
        NestedNormalizationGoals(vec![])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum Certainty {
    Yes,
    Maybe(MaybeCause),
}

impl Certainty {
    pub const AMBIGUOUS: Certainty = Certainty::Maybe(MaybeCause::Ambiguity);

    /// Use this function to merge the certainty of multiple nested subgoals.
    ///
    /// Given an impl like `impl<T: Foo + Bar> Baz for T {}`, we have 2 nested
    /// subgoals whenever we use the impl as a candidate: `T: Foo` and `T: Bar`.
    /// If evaluating `T: Foo` results in ambiguity and `T: Bar` results in
    /// success, we merge these two responses. This results in ambiguity.
    ///
    /// If we unify ambiguity with overflow, we return overflow. This doesn't matter
    /// inside of the solver as we do not distinguish ambiguity from overflow. It does
    /// however matter for diagnostics. If `T: Foo` resulted in overflow and `T: Bar`
    /// in ambiguity without changing the inference state, we still want to tell the
    /// user that `T: Baz` results in overflow.
    pub fn unify_with(self, other: Certainty) -> Certainty {
        match (self, other) {
            (Certainty::Yes, Certainty::Yes) => Certainty::Yes,
            (Certainty::Yes, Certainty::Maybe(_)) => other,
            (Certainty::Maybe(_), Certainty::Yes) => self,
            (Certainty::Maybe(a), Certainty::Maybe(b)) => Certainty::Maybe(a.unify_with(b)),
        }
    }

    pub const fn overflow(suggest_increasing_limit: bool) -> Certainty {
        Certainty::Maybe(MaybeCause::Overflow { suggest_increasing_limit })
    }
}

/// Why we failed to evaluate a goal.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum MaybeCause {
    /// We failed due to ambiguity. This ambiguity can either
    /// be a true ambiguity, i.e. there are multiple different answers,
    /// or we hit a case where we just don't bother, e.g. `?x: Trait` goals.
    Ambiguity,
    /// We gave up due to an overflow, most often by hitting the recursion limit.
    Overflow { suggest_increasing_limit: bool },
}

impl MaybeCause {
    fn unify_with(self, other: MaybeCause) -> MaybeCause {
        match (self, other) {
            (MaybeCause::Ambiguity, MaybeCause::Ambiguity) => MaybeCause::Ambiguity,
            (MaybeCause::Ambiguity, MaybeCause::Overflow { .. }) => other,
            (MaybeCause::Overflow { .. }, MaybeCause::Ambiguity) => self,
            (
                MaybeCause::Overflow { suggest_increasing_limit: a },
                MaybeCause::Overflow { suggest_increasing_limit: b },
            ) => MaybeCause::Overflow { suggest_increasing_limit: a || b },
        }
    }
}

#[derive_where(PartialEq, Eq, Debug; I: Interner)]
pub struct CacheData<I: Interner> {
    pub result: QueryResult<I>,
    pub proof_tree: Option<I::CanonicalGoalEvaluationStepRef>,
    pub additional_depth: usize,
    pub encountered_overflow: bool,
}
