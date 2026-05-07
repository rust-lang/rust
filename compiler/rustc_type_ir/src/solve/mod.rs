pub mod inspect;

use std::fmt::Debug;
use std::hash::Hash;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash, StableHash_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};
use tracing::debug;

use crate::lang_items::SolverTraitLangItem;
use crate::search_graph::PathKind;
use crate::{
    self as ty, Canonical, CanonicalVarValues, CantBeErased, Interner, TypingMode, Upcast,
};

pub type CanonicalInput<I, T = <I as Interner>::Predicate> =
    ty::CanonicalQueryInput<I, QueryInput<I, T>>;
pub type CanonicalResponse<I> = Canonical<I, Response<I>>;
/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<I> = Result<CanonicalResponse<I>, NoSolution>;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub struct NoSolution;

pub enum NoSolutionOrOpaquesAccessed {
    NoSolution(NoSolution),
    /// A bit like [`NoSolution`], but for functions that normally cannot fail *unless* they accessed
    /// opaues. (See [`TypingMode::ErasedNotCoherence`]). Getting `OpaquesAccessed` doesn't mean there
    /// truly is no solution. It just means that we want to bail out of the current query as fast as
    /// possible, possibly by returning `NoSolution` if that's fastest. This is okay because when you get
    /// `OpaquesAccessed` we're guaranteed that we're going to retry this query in the original typing
    /// mode to get the correct answer.
    OpaquesAccessed,
}

/// This conversion is sound, because even in we're in `OpaquesAccessed`,
/// we're going to retry so `NoSolution` is a valid response to give..
impl From<NoSolutionOrOpaquesAccessed> for NoSolution {
    fn from(
        (NoSolutionOrOpaquesAccessed::NoSolution(_) | NoSolutionOrOpaquesAccessed::OpaquesAccessed): NoSolutionOrOpaquesAccessed,
    ) -> Self {
        NoSolution
    }
}

impl From<NoSolution> for NoSolutionOrOpaquesAccessed {
    fn from(value: NoSolution) -> Self {
        Self::NoSolution(value)
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum SmallCopyList<T: Copy + Debug + Hash + Eq> {
    Empty,
    One([T; 1]),
    Two([T; 2]),
    Three([T; 3]),
}

impl<T: Copy + Debug + Hash + Eq> SmallCopyList<T> {
    fn empty() -> Self {
        Self::Empty
    }

    fn new(first: T) -> Self {
        Self::One([first])
    }

    /// Computes the union of two lists. Duplicates are removed.
    fn union(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Empty, other) | (other, Self::Empty) => Some(other),

            (Self::One([a]), Self::One([b])) if a == b => Some(Self::One([a])),
            (Self::One([a]), Self::One([b])) => Some(Self::Two([a, b])),
            (Self::One([a]), Self::Two([b, c])) | (Self::Two([a, b]), Self::One([c]))
                if a == b && b == c =>
            {
                Some(Self::One([a]))
            }
            (Self::One([a]), Self::Two([b, c])) | (Self::Two([a, b]), Self::One([c])) if a == b => {
                Some(Self::Two([a, c]))
            }
            (Self::One([a]), Self::Two([b, c])) | (Self::Two([a, b]), Self::One([c])) if a == c => {
                Some(Self::Two([a, b]))
            }
            (Self::One([a]), Self::Two([b, c])) | (Self::Two([a, b]), Self::One([c])) if b == c => {
                Some(Self::Two([a, b]))
            }
            (Self::One([a]), Self::Two([b, c])) | (Self::Two([a, b]), Self::One([c])) => {
                Some(Self::Three([a, b, c]))
            }
            _ => None,
        }
    }
}

impl<T: Copy + Debug + Hash + Eq> AsRef<[T]> for SmallCopyList<T> {
    fn as_ref(&self) -> &[T] {
        match self {
            Self::Empty => &[],
            Self::One(l) => l,
            Self::Two(l) => l,
            Self::Three(l) => l,
        }
    }
}

/// Information about how we accessed opaque types
/// This is what the trait solver does when each states is encountered:
///
/// |                         | bail? | rerun goal?                                                                                                          |
/// | ----------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- |
/// | never                   | no    | no                                                                                                                   |
/// | always                  | yes   | yes                                                                                                                  |
/// | [defid in storage]      | no    | only if any of the defids in the list is in the opaque type storage OR if TypingMode::PostAnalysis                   |
/// | opaque with hidden type | no    | only if any of the the opaques in the opaque type storage has a hidden type in this list AND if TypingMode::Analysis |
///
/// - "bail" is implemented with [`should_bail`](Self::should_bail).
///   If true, we're abandoning our attempt to canonicalize in [`TypingMode::ErasedNotCoherence`],
///   and should try to return as soon as possible to waste as little time as possible.
///   A rerun will be attempted in the original typing mode.
///
/// - Rerun goal is implemented with `should_rerun_after_erased_canonicalization`, on the `EvalCtxt`.
///
/// Some variant names contain an `Or` here. They rerun when any of the two conditions applies
#[derive_where(Copy, Clone, Debug, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum RerunCondition<I: Interner> {
    Never,

    /// Note that this only reruns according to the condition *if* we are in [`TypingMode::Analysis`].
    AnyOpaqueHasInferAsHidden,
    /// Note: unconditionally reruns in postanalysis
    OpaqueInStorage(SmallCopyList<I::LocalDefId>),

    /// Merges [`Self::AnyOpaqueHasInferAsHidden`] and [`Self::OpaqueInStorage`].
    /// Note that just like the unmerged [`Self::OpaqueInStorage`], that part of the
    /// condition only matters in [`TypingMode::Analysis`]
    OpaqueInStorageOrAnyOpaqueHasInferAsHidden(SmallCopyList<I::LocalDefId>),

    Always,
}

impl<I: Interner> RerunCondition<I> {
    /// Merge two rerun states according to the following transition diagram
    /// (some cells are empty because the table is symmetric, i.e. `a.merge(b)` == `b.merge(a)`).
    ///
    /// - "self" here means the current state, i.e. the state of the current column
    /// - square brackets represents that this is a list of things. Even if the state doesn't
    /// change, we might grow the list to effectively end up in a different state anyway
    /// - `[o. in s.]` abbreviates "opaque in storage"
    ///
    ///
    /// |                                 | never  | always | [opaque in storage] | opaque has infer as hidden | [o. in s.] or i. as hidden |
    /// | ------------------------------- | ------ | ------ | ------------------- | -------------------------- | -------------------------- |
    /// | never                           | self   | self   | self                | self                       | self                       |
    /// | always                          |        | always | always              | always                     | always                     |
    /// | [opaque in storage]             |        |        | concat self         | [o. in s.] or i. as hidden | concat to self             |
    /// | opaque has infer as hidden type |        |        |                     | self                       | to self                    |
    ///
    fn merge(self, other: Self) -> Self {
        let merged = match (self, other) {
            (Self::Never, other) | (other, Self::Never) => other,
            (Self::Always, _) | (_, Self::Always) => Self::Always,

            (Self::OpaqueInStorage(a), Self::OpaqueInStorage(b)) => {
                a.union(b).map(Self::OpaqueInStorage).unwrap_or(Self::Always)
            }
            (Self::AnyOpaqueHasInferAsHidden, Self::AnyOpaqueHasInferAsHidden) => {
                Self::AnyOpaqueHasInferAsHidden
            }
            (
                Self::AnyOpaqueHasInferAsHidden,
                Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(a),
            )
            | (
                Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(a),
                Self::AnyOpaqueHasInferAsHidden,
            ) => Self::OpaqueInStorage(a),

            (
                Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(a),
                Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(b),
            ) => a
                .union(b)
                .map(Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden)
                .unwrap_or(Self::Always),

            (Self::OpaqueInStorage(a), Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(b))
            | (Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(b), Self::OpaqueInStorage(a)) => a
                .union(b)
                .map(Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden)
                .unwrap_or(Self::Always),

            (Self::OpaqueInStorage(a), Self::AnyOpaqueHasInferAsHidden)
            | (Self::AnyOpaqueHasInferAsHidden, Self::OpaqueInStorage(a)) => {
                Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(a)
            }
        };
        debug!("merging rerun state {self:?} + {other:?} => {merged:?}");
        merged
    }

    #[must_use]
    fn should_bail(&self) -> bool {
        match self {
            Self::Always => true,
            Self::Never
            | Self::OpaqueInStorage(_)
            | Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(_)
            | Self::AnyOpaqueHasInferAsHidden => false,
        }
    }

    /// Returns true when any access of opaques was attempted.
    /// i.e. when `self != Self::Never`
    #[must_use]
    fn might_rerun(&self) -> bool {
        match self {
            Self::Never => false,
            Self::Always
            | Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(_)
            | Self::OpaqueInStorage(_)
            | Self::AnyOpaqueHasInferAsHidden => true,
        }
    }
}

/// Mainly for debugging, to keep track of the source of the rerunning
/// in [`TypingMode::ErasedNotCoherence`].
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum RerunReason {
    NormalizeOpaqueTypeRemoteCrate,
    NormalizeOpaqueType,
    MayUseUnstableFeature,
    EvaluateConst,
    SkipErasedAttempt,
    SelfTyInfer,
    FetchEligibleAssocItem,
    AutoTraitLeakage,
    TryStallCoroutine,
}

#[derive_where(Copy, Clone, Debug, Hash, PartialEq, Eq; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct AccessedOpaques<I: Interner> {
    #[cfg_attr(feature = "nightly", type_visitable(ignore))]
    #[type_foldable(identity)]
    pub reason: Option<RerunReason>,
    pub rerun: RerunCondition<I>,
}

impl<I: Interner> Default for AccessedOpaques<I> {
    fn default() -> Self {
        Self { reason: None, rerun: RerunCondition::Never }
    }
}

impl<I: Interner> AccessedOpaques<I> {
    pub fn update(&mut self, other: Self) {
        *self = Self {
            // prefer the newest reason
            reason: other.reason.or(self.reason),
            // merging accessed states can only result in MultipleOrUnknown
            rerun: self.rerun.merge(other.rerun),
        };
    }

    #[must_use]
    pub fn might_rerun(&self) -> bool {
        self.rerun.might_rerun()
    }

    #[must_use]
    pub fn should_bail(&self) -> bool {
        self.rerun.should_bail()
    }

    pub fn rerun_always(&mut self, reason: RerunReason) {
        debug!("set rerun always");
        self.update(AccessedOpaques { reason: Some(reason), rerun: RerunCondition::Always });
    }

    pub fn rerun_if_in_post_analysis(&mut self, reason: RerunReason) {
        debug!("set rerun if post analysis");
        self.update(AccessedOpaques {
            reason: Some(reason),
            rerun: RerunCondition::OpaqueInStorage(SmallCopyList::empty()),
        });
    }

    pub fn rerun_if_opaque_in_opaque_type_storage(
        &mut self,
        reason: RerunReason,
        defid: I::LocalDefId,
    ) {
        debug!("set rerun if opaque type {defid:?} in storage");
        self.update(AccessedOpaques {
            reason: Some(reason),
            rerun: RerunCondition::OpaqueInStorage(SmallCopyList::new(defid)),
        });
    }

    pub fn rerun_if_any_opaque_has_infer_as_hidden_type(&mut self, reason: RerunReason) {
        debug!("set rerun if any opaque in the storage has a hidden type that is an infer var");
        self.update(AccessedOpaques {
            reason: Some(reason),
            rerun: RerunCondition::AnyOpaqueHasInferAsHidden,
        });
    }
}

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, P)]
#[derive_where(Copy; I: Interner, P: Copy)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, StableHash_NoContext)
)]
pub struct Goal<I: Interner, P> {
    pub param_env: I::ParamEnv,
    pub predicate: P,
}

impl<I: Interner, P: Eq> Eq for Goal<I, P> {}

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
/// their source. This is used to decide whether a cycle is coinductive.
/// See the documentation of `EvalCtxt::step_kind_for_source` for more details
/// about this.
///
/// It is also used by proof tree visitors, e.g. for diagnostics purposes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub enum GoalSource {
    Misc,
    /// A nested goal required to prove that types are equal/subtypes.
    /// This is always an unproductive step.
    ///
    /// This is also used for all `NormalizesTo` goals as we they are used
    /// to relate types in `AliasRelate`.
    TypeRelating,
    /// We're proving a where-bound of an impl.
    ImplWhereBound,
    /// Const conditions that need to hold for `[const]` alias bounds to hold.
    AliasBoundConstCondition,
    /// Predicate required for an alias projection to be well-formed.
    /// This is used in three places:
    /// 1. projecting to an opaque whose hidden type is already registered in
    ///    the opaque type storage,
    /// 2. for rigid projections's trait goal,
    /// 3. for GAT where clauses.
    AliasWellFormed,
    /// In case normalizing aliases in nested goals cycles, eagerly normalizing these
    /// aliases in the context of the parent may incorrectly change the cycle kind.
    /// Normalizing aliases in goals therefore tracks the original path kind for this
    /// nested goal. See the comment of the `ReplaceAliasWithInfer` visitor for more
    /// details.
    NormalizeGoal(PathKind),
}

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, Goal<I, P>)]
#[derive_where(Copy; I: Interner, Goal<I, P>: Copy)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, StableHash_NoContext)
)]
pub struct QueryInput<I: Interner, P> {
    pub goal: Goal<I, P>,
    pub predefined_opaques_in_body: I::PredefinedOpaques,
}

impl<I: Interner, P: Eq> Eq for QueryInput<I, P> {}

/// Which trait candidates should be preferred over other candidates? By default, prefer where
/// bounds over alias bounds. For marker traits, prefer alias bounds over where bounds.
#[derive(Clone, Copy, Debug)]
pub enum CandidatePreferenceMode {
    /// Prefers where bounds over alias bounds
    Default,
    /// Prefers alias bounds over where bounds
    Marker,
}

impl CandidatePreferenceMode {
    /// Given `trait_def_id`, which candidate preference mode should be used?
    pub fn compute<I: Interner>(cx: I, trait_id: I::TraitId) -> CandidatePreferenceMode {
        let is_sizedness_or_auto_or_default_goal = cx.is_sizedness_trait(trait_id)
            || cx.trait_is_auto(trait_id)
            || cx.is_default_trait(trait_id);
        if is_sizedness_or_auto_or_default_goal {
            CandidatePreferenceMode::Marker
        } else {
            CandidatePreferenceMode::Default
        }
    }
}

/// Possible ways the given goal can be proven.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
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
    Impl(I::ImplId),
    /// A builtin impl generated by the compiler. When adding a new special
    /// trait, try to use actual impls whenever possible. Builtin impls should
    /// only be used in cases where the impl cannot be manually be written.
    ///
    /// Notable examples are auto traits, `Sized`, and `DiscriminantKind`.
    /// For a list of all traits with builtin impls, check out the
    /// `EvalCtxt::assemble_builtin_impl_candidates` method.
    BuiltinImpl(BuiltinImplSource),
    /// An assumption from the environment. Stores a [`ParamEnvSource`], since we
    /// prefer non-global param-env candidates in candidate assembly.
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
    ParamEnv(ParamEnvSource),
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
    AliasBound(AliasBoundKind),
    /// A candidate that is registered only during coherence to represent some
    /// yet-unknown impl that could be produced downstream without violating orphan
    /// rules.
    // FIXME: Merge this with the forced ambiguity candidates, so those don't use `Misc`.
    CoherenceUnknowable,
}

impl<I: Interner> Eq for CandidateSource<I> {}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum ParamEnvSource {
    /// Preferred eagerly.
    NonGlobal,
    // Not considered unless there are non-global param-env candidates too.
    Global,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
pub enum AliasBoundKind {
    /// Alias bound from the self type of a projection
    SelfBounds,
    // Alias bound having recursed on the self type of a projection
    NonSelfBounds,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash, Encodable_NoContext, Decodable_NoContext))]
pub enum BuiltinImplSource {
    /// A built-in impl that is considered trivial, without any nested requirements. They
    /// are preferred over where-clauses, and we want to track them explicitly.
    Trivial,
    /// Some built-in impl we don't need to differentiate. This should be used
    /// unless more specific information is necessary.
    Misc,
    /// A built-in impl for trait objects. The index is only used in winnowing.
    Object(usize),
    /// A built-in implementation of `Upcast` for trait objects to other trait objects.
    ///
    /// The index is only used for winnowing.
    TraitUpcasting(usize),
}

#[derive_where(Copy, Clone, Debug; I: Interner)]
pub enum FetchEligibleAssocItemResponse<I: Interner> {
    Err(I::ErrorGuaranteed),
    Found(I::ImplOrTraitAssocTermId),
    NotFound(TypingMode<I, CantBeErased>),
    NotFoundBecauseErased,
}

#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct Response<I: Interner> {
    pub certainty: Certainty,
    pub var_values: CanonicalVarValues<I>,
    /// Additional constraints returned by this query.
    pub external_constraints: I::ExternalConstraints,
}

impl<I: Interner> Eq for Response<I> {}

/// Additional constraints returned on success.
#[derive_where(Clone, Hash, PartialEq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct ExternalConstraintsData<I: Interner> {
    pub region_constraints: Vec<(ty::RegionConstraint<I>, VisibleForLeakCheck)>,
    pub opaque_types: Vec<(ty::OpaqueTypeKey<I>, I::Ty)>,
    pub normalization_nested_goals: NestedNormalizationGoals<I>,
}

impl<I: Interner> Eq for ExternalConstraintsData<I> {}

impl<I: Interner> ExternalConstraintsData<I> {
    pub fn is_empty(&self) -> bool {
        self.region_constraints.is_empty()
            && self.opaque_types.is_empty()
            && self.normalization_nested_goals.is_empty()
    }
}

/// Whether the given region constraint should be considered/ignored for
/// leak check. In most part of the compiler, this should be `Yes`, except
/// for applying constraints from the nested goals in next-solver.
/// `Unreachable` is used in places in which leak check isn't done, e.g.
/// borrowck.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum VisibleForLeakCheck {
    Yes,
    No,
    Unreachable,
}

impl VisibleForLeakCheck {
    pub fn and(self, other: VisibleForLeakCheck) -> VisibleForLeakCheck {
        match (self, other) {
            // Make sure that we never overwrite that constraints shouldn't
            // be encountered by the leak checked
            (VisibleForLeakCheck::Unreachable, _) | (_, VisibleForLeakCheck::Unreachable) => {
                VisibleForLeakCheck::Unreachable
            }
            (VisibleForLeakCheck::No, _) | (_, VisibleForLeakCheck::No) => VisibleForLeakCheck::No,
            (VisibleForLeakCheck::Yes, VisibleForLeakCheck::Yes) => VisibleForLeakCheck::Yes,
        }
    }

    pub fn or(self, other: VisibleForLeakCheck) -> VisibleForLeakCheck {
        match (self, other) {
            // Make sure that we never overwrite that constraints shouldn't
            // be encountered by the leak checked
            (VisibleForLeakCheck::Unreachable, _) | (_, VisibleForLeakCheck::Unreachable) => {
                VisibleForLeakCheck::Unreachable
            }
            (VisibleForLeakCheck::Yes, _) | (_, VisibleForLeakCheck::Yes) => {
                VisibleForLeakCheck::Yes
            }
            (VisibleForLeakCheck::No, VisibleForLeakCheck::No) => VisibleForLeakCheck::No,
        }
    }
}

#[derive_where(Clone, Hash, PartialEq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct NestedNormalizationGoals<I: Interner>(pub Vec<(GoalSource, Goal<I, I::Predicate>)>);

impl<I: Interner> Eq for NestedNormalizationGoals<I> {}

impl<I: Interner> NestedNormalizationGoals<I> {
    pub fn empty() -> Self {
        NestedNormalizationGoals(vec![])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub enum Certainty {
    Yes,
    Maybe(MaybeInfo),
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct MaybeInfo {
    pub cause: MaybeCause,
    pub opaque_types_jank: OpaqueTypesJank,
    pub stalled_on_coroutines: StalledOnCoroutines,
}

impl MaybeInfo {
    pub const AMBIGUOUS: MaybeInfo = MaybeInfo {
        cause: MaybeCause::Ambiguity,
        opaque_types_jank: OpaqueTypesJank::AllGood,
        stalled_on_coroutines: StalledOnCoroutines::No,
    };

    fn and(self, other: MaybeInfo) -> MaybeInfo {
        MaybeInfo {
            cause: self.cause.and(other.cause),
            opaque_types_jank: self.opaque_types_jank.and(other.opaque_types_jank),
            stalled_on_coroutines: self.stalled_on_coroutines.and(other.stalled_on_coroutines),
        }
    }

    pub fn or(self, other: MaybeInfo) -> MaybeInfo {
        MaybeInfo {
            cause: self.cause.or(other.cause),
            opaque_types_jank: self.opaque_types_jank.or(other.opaque_types_jank),
            stalled_on_coroutines: self.stalled_on_coroutines.or(other.stalled_on_coroutines),
        }
    }
}

/// Supporting not-yet-defined opaque types in HIR typeck is somewhat
/// challenging. Ideally we'd normalize them to a new inference variable
/// and just defer type inference which relies on the opaque until we've
/// constrained the hidden type.
///
/// This doesn't work for method and function calls as we need to guide type
/// inference for the function arguments. We treat not-yet-defined opaque types
/// as if they were rigid instead in these places.
///
/// When we encounter a `?hidden_type_of_opaque: Trait<?var>` goal, we use the
/// item bounds and blanket impls to guide inference by constraining other type
/// variables, see `EvalCtxt::try_assemble_bounds_via_registered_opaques`. We
/// always keep the certainty as `Maybe` so that we properly prove these goals
/// once the hidden type has been constrained.
///
/// If we fail to prove the trait goal via item bounds or blanket impls, the
/// goal would have errored if the opaque type were rigid. In this case, we
/// set `OpaqueTypesJank::ErrorIfRigidSelfTy` in the [Certainty].
///
/// Places in HIR typeck where we want to treat not-yet-defined opaque types as if
/// they were kind of rigid then use `fn root_goal_may_hold_opaque_types_jank` which
/// returns `false` if the goal doesn't hold or if `OpaqueTypesJank::ErrorIfRigidSelfTy`
/// is set (i.e. proving it required relies on some `?hidden_ty: NotInItemBounds` goal).
///
/// This is subtly different from actually treating not-yet-defined opaque types as
/// rigid, e.g. it allows constraining opaque types if they are not the self-type of
/// a goal. It is good enough for now and only matters for very rare type inference
/// edge cases. We can improve this later on if necessary.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub enum OpaqueTypesJank {
    AllGood,
    ErrorIfRigidSelfTy,
}
impl OpaqueTypesJank {
    fn and(self, other: OpaqueTypesJank) -> OpaqueTypesJank {
        match (self, other) {
            (OpaqueTypesJank::AllGood, OpaqueTypesJank::AllGood) => OpaqueTypesJank::AllGood,
            (OpaqueTypesJank::ErrorIfRigidSelfTy, _) | (_, OpaqueTypesJank::ErrorIfRigidSelfTy) => {
                OpaqueTypesJank::ErrorIfRigidSelfTy
            }
        }
    }

    pub fn or(self, other: OpaqueTypesJank) -> OpaqueTypesJank {
        match (self, other) {
            (OpaqueTypesJank::ErrorIfRigidSelfTy, OpaqueTypesJank::ErrorIfRigidSelfTy) => {
                OpaqueTypesJank::ErrorIfRigidSelfTy
            }
            (OpaqueTypesJank::AllGood, _) | (_, OpaqueTypesJank::AllGood) => {
                OpaqueTypesJank::AllGood
            }
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum StalledOnCoroutines {
    Yes,
    No,
}

impl StalledOnCoroutines {
    fn and(self, other: StalledOnCoroutines) -> StalledOnCoroutines {
        match (self, other) {
            (StalledOnCoroutines::No, StalledOnCoroutines::No) => StalledOnCoroutines::No,
            (StalledOnCoroutines::Yes, _) | (_, StalledOnCoroutines::Yes) => {
                StalledOnCoroutines::Yes
            }
        }
    }

    pub fn or(self, other: StalledOnCoroutines) -> StalledOnCoroutines {
        // `StalledOnCoroutines::Yes` is contagious: obtaining `Certainty::Maybe`
        // while a candidate is stalled on a coroutine might have been
        // `Certainty::Yes` or `NoSolution` if it were not stalled.
        StalledOnCoroutines::and(self, other)
    }
}

impl Certainty {
    pub const AMBIGUOUS: Certainty = Certainty::Maybe(MaybeInfo::AMBIGUOUS);

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
    pub fn and(self, other: Certainty) -> Certainty {
        match (self, other) {
            (Certainty::Yes, Certainty::Yes) => Certainty::Yes,
            (Certainty::Yes, Certainty::Maybe { .. }) => other,
            (Certainty::Maybe { .. }, Certainty::Yes) => self,
            (Certainty::Maybe(a_maybe), Certainty::Maybe(b_maybe)) => {
                Certainty::Maybe(a_maybe.and(b_maybe))
            }
        }
    }

    pub const fn overflow(suggest_increasing_limit: bool) -> Certainty {
        Certainty::Maybe(MaybeInfo {
            cause: MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: false },
            opaque_types_jank: OpaqueTypesJank::AllGood,
            stalled_on_coroutines: StalledOnCoroutines::No,
        })
    }
}

/// Why we failed to evaluate a goal.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub enum MaybeCause {
    /// We failed due to ambiguity. This ambiguity can either
    /// be a true ambiguity, i.e. there are multiple different answers,
    /// or we hit a case where we just don't bother, e.g. `?x: Trait` goals.
    Ambiguity,
    /// We gave up due to an overflow, most often by hitting the recursion limit.
    Overflow { suggest_increasing_limit: bool, keep_constraints: bool },
}

impl MaybeCause {
    fn and(self, other: MaybeCause) -> MaybeCause {
        match (self, other) {
            (MaybeCause::Ambiguity, MaybeCause::Ambiguity) => MaybeCause::Ambiguity,
            (MaybeCause::Ambiguity, MaybeCause::Overflow { .. }) => other,
            (MaybeCause::Overflow { .. }, MaybeCause::Ambiguity) => self,
            (
                MaybeCause::Overflow {
                    suggest_increasing_limit: limit_a,
                    keep_constraints: keep_a,
                },
                MaybeCause::Overflow {
                    suggest_increasing_limit: limit_b,
                    keep_constraints: keep_b,
                },
            ) => MaybeCause::Overflow {
                suggest_increasing_limit: limit_a && limit_b,
                keep_constraints: keep_a && keep_b,
            },
        }
    }

    pub fn or(self, other: MaybeCause) -> MaybeCause {
        match (self, other) {
            (MaybeCause::Ambiguity, MaybeCause::Ambiguity) => MaybeCause::Ambiguity,

            // When combining ambiguity + overflow, we can keep constraints.
            (
                MaybeCause::Ambiguity,
                MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: _ },
            ) => MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: true },
            (
                MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: _ },
                MaybeCause::Ambiguity,
            ) => MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: true },

            (
                MaybeCause::Overflow {
                    suggest_increasing_limit: limit_a,
                    keep_constraints: keep_a,
                },
                MaybeCause::Overflow {
                    suggest_increasing_limit: limit_b,
                    keep_constraints: keep_b,
                },
            ) => MaybeCause::Overflow {
                suggest_increasing_limit: limit_a || limit_b,
                keep_constraints: keep_a || keep_b,
            },
        }
    }
}

/// Indicates that a `impl Drop for Adt` is `const` or not.
#[derive(Debug)]
pub enum AdtDestructorKind {
    NotConst,
    Const,
}

/// Which sizedness trait - `Sized`, `MetaSized`? `PointeeSized` is omitted as it is removed during
/// lowering.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub enum SizedTraitKind {
    /// `Sized` trait
    Sized,
    /// `MetaSized` trait
    MetaSized,
}

impl SizedTraitKind {
    /// Returns `DefId` of corresponding language item.
    pub fn require_lang_item<I: Interner>(self, cx: I) -> I::TraitId {
        cx.require_trait_lang_item(match self {
            SizedTraitKind::Sized => SolverTraitLangItem::Sized,
            SizedTraitKind::MetaSized => SolverTraitLangItem::MetaSized,
        })
    }
}
