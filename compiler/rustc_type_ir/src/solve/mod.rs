pub mod inspect;

use std::convert::Infallible;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Deref;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash, StableHash_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};
use thin_vec::ThinVec;
use tracing::debug;

use crate::inherent::*;
use crate::lang_items::SolverTraitLangItem;
use crate::region_constraint::RegionConstraint;
use crate::search_graph::PathKind;
use crate::{
    self as ty, BoundEvidence, BoundVarIndexKind, Canonical, CanonicalVarValues, CantBeErased,
    ConstVid, FallibleTypeFolder, FloatVid, GenericArgKind, InferConst, IntVid, Interner,
    PlaceholderEvidence, TermKind, TyVid, TypeFoldable, TypeFolder, TypeVisitable, TypeVisitor,
    TypingMode, Upcast,
};

pub type CanonicalInputData<I> =
    ty::CanonicalQueryInput<I, QueryInput<I, <I as Interner>::Predicate>>;
pub type CanonicalResponse<I> = Canonical<I, Response<I>>;
/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<I> = Result<CanonicalResponse<I>, NoSolution>;
pub type QueryResultOrRerunNonErased<I> = Result<CanonicalResponse<I>, NoSolutionOrRerunNonErased>;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub struct NoSolution;

pub trait RerunResultExt<T> {
    fn map_err_to_rerun(self) -> Result<Result<T, NoSolution>, RerunNonErased>;
}

impl<T> RerunResultExt<T> for Result<T, NoSolutionOrRerunNonErased> {
    fn map_err_to_rerun(self) -> Result<Result<T, NoSolution>, RerunNonErased> {
        match self {
            Ok(i) => Ok(Ok(i)),
            Err(NoSolutionOrRerunNonErased::NoSolution(NoSolution)) => Ok(Err(NoSolution)),
            Err(NoSolutionOrRerunNonErased::RerunNonErased(e)) => Err(e),
        }
    }
}

/// A bit like [`NoSolution`], but for functions that normally cannot fail *unless* they accessed
/// opaues. (See [`TypingMode::ErasedNotCoherence`]). Getting `OpaquesAccessed` doesn't mean there
/// truly is no solution. It just means that we want to bail out of the current query as fast as
/// possible, possibly by returning `NoSolution` if that's fastest. This is okay because when you get
/// `OpaquesAccessed` we're guaranteed that we're going to retry this query in the original typing
/// mode to get the correct answer.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub struct RerunNonErased(());

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "nightly", derive(StableHash))]
pub enum NoSolutionOrRerunNonErased {
    NoSolution(NoSolution),
    RerunNonErased(RerunNonErased),
}

impl From<NoSolution> for NoSolutionOrRerunNonErased {
    fn from(value: NoSolution) -> Self {
        Self::NoSolution(value)
    }
}

impl From<RerunNonErased> for NoSolutionOrRerunNonErased {
    fn from(value: RerunNonErased) -> Self {
        Self::RerunNonErased(value)
    }
}

/// A small set of up to 3 `Copy` elements, used as an optimization in [`RerunCondition`].
/// The entire set can be `Copy`ed because of this requirement.
///
/// Set properties maintained using [`union`](SmallCopySet::union), which deduplicates values.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum SmallCopySet<T: Copy + Debug + Hash + Eq> {
    Empty,
    One([T; 1]),
    Two([T; 2]),
    Three([T; 3]),
}

impl<T: Copy + Debug + Hash + Eq> SmallCopySet<T> {
    fn empty() -> Self {
        Self::Empty
    }

    fn new(first: T) -> Self {
        Self::One([first])
    }

    /// Computes the union of two lists. Duplicates are removed.
    ///
    /// Since the set can hold at most 3 elements, returns `None` if the resulting set cannot be
    /// represented.
    ///
    /// In the context of [`RerunCondition`], this means we fall back to rerunning unconditionally.
    /// This can be beneficial, since at some point, tracking all the conditions under which a query
    /// has to be rerun becomes slower than just rerunning unconditionally. This is especially so,
    /// since as long as rerun conditions are tracked, we keep executing the current query. As soon as
    /// we cannot track anymore, and unconditionally rerun, we also abort the current query.
    /// By at some point opting to abort early, we may save a lot of time skipping further work
    /// that will have to likely be redone anyway.
    ///
    /// note that *not* all cases are handled. you can union two lists of two elements with equal
    /// elements, and still get `none` back. checking for all cases is more work than just rerunning
    /// in some cases.
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
            // There are some more cases we could handle, like 2 + 2 => 3 if there's one duplicate,
            // But the check seems to be more expensive than the gain. Even then, the difference is
            // tiny, and could just be noise. Not worth it regardless.
            _ => None,
        }
    }
}

impl<T: Copy + Debug + Hash + Eq> AsRef<[T]> for SmallCopySet<T> {
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
/// | opaque with hidden type | no    | only if any of the opaques in the opaque type storage has a hidden type in this list AND if TypingMode::Typeck       |
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

    /// Note that this only reruns according to the condition *if* we are in [`TypingMode::Typeck`].
    AnyOpaqueHasInferAsHidden,
    /// Note: unconditionally reruns in postanalysis
    OpaqueInStorage(SmallCopySet<I::LocalDefId>),

    /// Merges [`Self::AnyOpaqueHasInferAsHidden`] and [`Self::OpaqueInStorage`].
    /// Note that just like the unmerged [`Self::OpaqueInStorage`], that part of the
    /// condition only matters in [`TypingMode::Typeck`]
    OpaqueInStorageOrAnyOpaqueHasInferAsHidden(SmallCopySet<I::LocalDefId>),

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
    fn should_bail(&self) -> Result<(), RerunNonErased> {
        match self {
            Self::Always => Err(RerunNonErased(())),
            Self::Never
            | Self::OpaqueInStorage(_)
            | Self::OpaqueInStorageOrAnyOpaqueHasInferAsHidden(_)
            | Self::AnyOpaqueHasInferAsHidden => Ok(()),
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
    pub fn update(&mut self, other: Self) -> Result<(), RerunNonErased> {
        *self = Self {
            // prefer the newest reason
            reason: other.reason.or(self.reason),
            // merging accessed states can only result in MultipleOrUnknown
            rerun: self.rerun.merge(other.rerun),
        };

        self.should_bail()
    }

    #[must_use]
    pub fn might_rerun(&self) -> bool {
        self.rerun.might_rerun()
    }

    #[must_use]
    pub fn should_bail(&self) -> Result<(), RerunNonErased> {
        self.rerun.should_bail()
    }

    pub fn rerun_always(&mut self, reason: RerunReason) -> Result<Infallible, RerunNonErased> {
        debug!("set rerun always");
        match self.update(AccessedOpaques { reason: Some(reason), rerun: RerunCondition::Always }) {
            Ok(_) => unreachable!(),
            Err(e) => Err(e),
        }
    }

    pub fn rerun_if_in_post_analysis(&mut self, reason: RerunReason) -> Result<(), RerunNonErased> {
        debug!("set rerun if post analysis");
        self.update(AccessedOpaques {
            reason: Some(reason),
            rerun: RerunCondition::OpaqueInStorage(SmallCopySet::empty()),
        })
    }

    pub fn rerun_if_opaque_in_opaque_type_storage(
        &mut self,
        reason: RerunReason,
        defid: I::LocalOpaqueTyId,
    ) -> Result<(), RerunNonErased> {
        debug!("set rerun if opaque type {defid:?} in storage");
        self.update(AccessedOpaques {
            reason: Some(reason),
            rerun: RerunCondition::OpaqueInStorage(SmallCopySet::new(defid.into())),
        })
    }

    pub fn rerun_if_any_opaque_has_infer_as_hidden_type(
        &mut self,
        reason: RerunReason,
    ) -> Result<(), RerunNonErased> {
        debug!("set rerun if any opaque in the storage has a hidden type that is an infer var");
        self.update(AccessedOpaques {
            reason: Some(reason),
            rerun: RerunCondition::AnyOpaqueHasInferAsHidden,
        })
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
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash, Encodable_NoContext, Decodable_NoContext))]
pub enum ParamEnvSource {
    /// Preferred eagerly.
    NonGlobal,
    // Not considered unless there are non-global param-env candidates too.
    Global,
}

/// Source identity shared by the principal trait clause and every associated
/// equality written in the same HIR trait bound.
///
/// The local id is scoped by `owner`; it is never interpreted without that
/// owner and therefore remains stable across clause elaboration and reordering.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct ItemContractKey<I: Interner> {
    pub owner: I::DefId,
    #[lift(identity)]
    pub hir_local_id: u32,
}

impl<I: Interner> Eq for ItemContractKey<I> {}

/// One source contract instantiated with all early arguments of its owner.
///
/// Keeping the complete substitution in the identity prevents two inherited
/// uses of the same source bound from sharing proof identity accidentally.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct InstantiatedItemContract<I: Interner> {
    pub key: ItemContractKey<I>,
    pub complete_early_args: I::GenericArgs,
}

impl<I: Interner> Eq for InstantiatedItemContract<I> {}

/// A source dictionary contract instantiated in one variable scope.
///
/// The identity records the source bound and its complete early substitution.
/// The clauses retain the principal trait clause, associated equalities, and
/// host-effect requirements as one bundle. `ordinary_args` records the binder
/// substitution used to open the contract.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct RequiredContract<I: Interner>(pub I::BoundRequiredContract);

impl<I: Interner> Eq for RequiredContract<I> {}

impl<I: Interner> Deref for RequiredContract<I> {
    type Target = ty::BoundRequiredContractData<I>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<I: Interner> RequiredContract<I> {
    pub fn new(
        cx: I,
        identity: InstantiatedItemContract<I>,
        clauses: I::Clauses,
        principal_index: u32,
        ordinary_args: Option<I::GenericArgs>,
    ) -> Self {
        assert!(
            clauses.get(principal_index as usize).is_some(),
            "required-contract principal index is out of bounds",
        );
        RequiredContract(cx.mk_bound_required_contract(ty::BoundRequiredContractData {
            identity,
            clauses,
            principal_index,
            ordinary_args,
        }))
    }

    pub fn principal_clause(self) -> I::Clause {
        self.clauses
            .get(self.principal_index as usize)
            .expect("required-contract principal index is out of bounds")
    }

    pub fn is_principal_clause(self, clause: I::Clause) -> bool {
        self.principal_clause() == clause
    }
}

/// Stable semantic identity of one clause in a parameter environment.
///
/// Origins distinguish positional caller bounds, item-owned clauses, and
/// binder-owned assumptions. Source contracts carry their complete early
/// substitution alongside the source key.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub enum ParamEnvAssumption<I: Interner> {
    CallerBound {
        #[lift(identity)]
        index: u32,
    },
    /// Clause declared on a concrete item. The owner plus the index in that
    /// item's own clause list stays stable when inherited clauses are rebased
    /// or a parameter environment is rebuilt with a different prefix.
    ItemClause {
        owner: I::DefId,
        #[lift(identity)]
        index: u32,
    },
    /// A source-written trait bound and all clauses derived from its single
    /// dictionary contract. Unlike `ItemClause`, this identity is independent
    /// of clause ordering and survives supertrait elaboration.
    ItemContract { contract: InstantiatedItemContract<I> },
    /// Compiler-generated clause attached to an item (for example an RPITIT
    /// equality or a const condition). These use a separate index namespace
    /// from user-written item clauses.
    Generated {
        owner: I::DefId,
        #[lift(identity)]
        index: u32,
    },
    Binder {
        #[lift(identity)]
        telescope_index: u32,
        identity: I::Clause,
        instantiation: I::GenericArgs,
    },
}

impl<I: Interner> Eq for ParamEnvAssumption<I> {}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash, Encodable_NoContext, Decodable_NoContext))]
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
    // FIXME(-Znext-solver=no): The new solver does not need this index, remove!
    Object(usize),
    /// A built-in implementation of `Upcast` for trait objects to other trait objects.
    ///
    /// The index is only used for winnowing.
    // FIXME(-Znext-solver=no): The new solver does not need this index, remove!
    TraitUpcasting(usize),
}

/// Data attached to a builtin proof node.
///
/// Callable proof nodes retain their output and ordinary binder substitution
/// together, preserving the relationship between their arguments and result.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub enum BuiltinEvidence<I: Interner> {
    /// The builtin rule needs no proof-local data beyond the proven trait ref.
    RuleOnly,
    /// The output and binder substitution of an `Fn`, `FnMut`, or `FnOnce`
    /// proof. The instantiated inputs are in the owning node's trait ref;
    /// the substitution also retains variables absent from those inputs.
    Fn { output: I::Ty, instantiation: I::GenericArgs },
    /// Awaited output and the input binder substitution of an async callable.
    AsyncFn { output: I::Ty, instantiation: I::GenericArgs },
}

impl<I: Interner> Eq for BuiltinEvidence<I> {}

/// One call-operation instantiation of an output-only projection bound carried
/// by a trait object. The bound retains its declarations, and `ordinary_args`
/// records their ordinary substitution separately from the owning Dyn evidence.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct DynProjectionOperation<I: Interner> {
    pub projection_bound: I::Clause,
    pub ordinary_args: I::GenericArgs,
}

impl<I: Interner> Eq for DynProjectionOperation<I> {}

/// Stable identity of the rule selected for one node in a trait proof.
///
/// Unlike [`CandidateSource`], this also identifies a specific param-env
/// assumption. Its type and const payloads use the enclosing proof's variable
/// scope and participate in substitution with the proven trait reference.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub enum CandidateEvidenceSource<I: Interner> {
    /// Multiple proof paths were explicitly quotiented by coherence. The
    /// selected concrete recipe is the sole nested node of this proof node.
    Unique(CoherenceKey<I>),
    Impl {
        impl_def_id: I::ImplId,
        args: I::GenericArgs,
    },
    Builtin {
        source: BuiltinImplSource,
        evidence: BuiltinEvidence<I>,
    },
    /// Proof obtained from a bound carried by a trait object. The bound is
    /// stored in binder-preserving clause form, with an optional stable vtable slot.
    Dyn {
        object_bound: I::Clause,
        /// The exact ordinary binder substitution, or `None` for a proof
        /// independent of every ordinary binder variable.
        instantiation: Option<I::GenericArgs>,
        vtable_slot: Option<u32>,
        operation: Option<DynProjectionOperation<I>>,
    },
    ParamEnv {
        source: ParamEnvSource,
        origin: ParamEnvAssumption<I>,
    },
    AliasBound(AliasBoundKind),
    /// Error-recovery data, which cannot establish proof identity.
    Error,
    CoherenceUnknowable,
}

impl<I: Interner> Eq for CandidateEvidenceSource<I> {}

/// Stable identity used when coherence proves that candidate choice is not
/// semantically observable for a trait ref.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct CoherenceKey<I: Interner> {
    pub trait_ref: ty::TraitRef<I>,
}

impl<I: Interner> Eq for CoherenceKey<I> {}

impl<I: Interner> CandidateEvidenceSource<I> {
    /// Converts source tags that need no additional proof payload.
    /// Impl, param-env, and alias-bound tags lack the arguments or origin
    /// required by their evidence representation and return `None`.
    pub fn from_source(source: CandidateSource<I>) -> Option<Self> {
        match source {
            CandidateSource::Impl(_)
            | CandidateSource::ParamEnv(_)
            | CandidateSource::AliasBound(_) => None,
            CandidateSource::BuiltinImpl(source) => Some(CandidateEvidenceSource::Builtin {
                source,
                evidence: BuiltinEvidence::RuleOnly,
            }),
            CandidateSource::CoherenceUnknowable => {
                Some(CandidateEvidenceSource::CoherenceUnknowable)
            }
        }
    }
}

/// One node in a trait proof DAG.
///
/// `nested` contains stable indices into the owning [`CandidateEvidence`]. A
/// node stores the instantiated trait ref it proves so that consumers can
/// validate that a recipe is not accidentally reused for a different goal.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct CandidateEvidenceNode<I: Interner> {
    pub trait_ref: ty::TraitRef<I>,
    pub source: CandidateEvidenceSource<I>,
    /// Edges to nodes in the same variable scope. This is used for
    /// wrappers such as [`CandidateEvidenceSource::Unique`].
    pub nested: Vec<u32>,
    /// Interned nested proofs in this node's variable scope.
    pub nested_evidence: Vec<CandidateEvidenceUse<I>>,
}

impl<I: Interner> Eq for CandidateEvidenceNode<I> {}

/// A proof recipe for a trait goal.
///
/// Nodes in one variable scope use a flat DAG with interned nested proofs.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct CandidateEvidence<I: Interner> {
    pub root: u32,
    pub nodes: Vec<CandidateEvidenceNode<I>>,
}

impl<I: Interner> Eq for CandidateEvidence<I> {}

impl<I: Interner> CandidateEvidence<I> {
    /// Whether this proof and all of its nested proofs are structural builtin
    /// rules which carry no associated value or dictionary choice.
    pub fn is_rule_only(&self) -> bool {
        self.nodes.iter().all(|node| {
            matches!(
                node.source,
                CandidateEvidenceSource::Builtin { evidence: BuiltinEvidence::RuleOnly, .. }
                    | CandidateEvidenceSource::Unique(_)
            ) && node.nested_evidence.iter().all(CandidateEvidenceUse::is_rule_only)
        })
    }

    /// Returns the equivalence classes of proof nodes whose trait refs are one
    /// semantic identity. A coherence `Unique` node and the concrete recipe it
    /// wraps deliberately repeat that identity in the serialized DAG, but a
    /// type folder must not instantiate the repeated occurrences separately.
    fn trait_ref_classes(&self) -> Vec<usize> {
        fn find(parents: &mut [usize], mut index: usize) -> usize {
            while parents[index] != index {
                let parent = parents[index];
                parents[index] = parents[parent];
                index = parents[index];
            }
            index
        }

        let mut parents = (0..self.nodes.len()).collect::<Vec<_>>();
        for (index, node) in self.nodes.iter().enumerate() {
            if matches!(node.source, CandidateEvidenceSource::Unique(_)) {
                let nested = usize::try_from(node.nested[0]).expect("proof node index overflow");
                let index_root = find(&mut parents, index);
                let nested_root = find(&mut parents, nested);
                if index_root != nested_root {
                    parents[nested_root] = index_root;
                }
            }
        }
        for index in 0..parents.len() {
            parents[index] = find(&mut parents, index);
        }
        parents
    }
}

impl<I: Interner> TypeFoldable<I> for CandidateEvidence<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        let classes = self.trait_ref_classes();
        let mut folded_trait_refs = vec![None; self.nodes.len()];
        let mut nodes = Vec::with_capacity(self.nodes.len());

        for (index, node) in self.nodes.into_iter().enumerate() {
            let class = classes[index];
            let trait_ref = match folded_trait_refs[class] {
                Some(trait_ref) => trait_ref,
                None => {
                    let trait_ref = node.trait_ref.try_fold_with(folder)?;
                    folded_trait_refs[class] = Some(trait_ref);
                    trait_ref
                }
            };
            let source = match node.source {
                CandidateEvidenceSource::Unique(_) => {
                    CandidateEvidenceSource::Unique(CoherenceKey { trait_ref })
                }
                source => source.try_fold_with(folder)?,
            };
            nodes.push(CandidateEvidenceNode {
                trait_ref,
                source,
                nested: node.nested,
                nested_evidence: node.nested_evidence.try_fold_with(folder)?,
            });
        }

        Ok(CandidateEvidence { root: self.root, nodes })
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        let classes = self.trait_ref_classes();
        let mut folded_trait_refs = vec![None; self.nodes.len()];
        let mut nodes = Vec::with_capacity(self.nodes.len());

        for (index, node) in self.nodes.into_iter().enumerate() {
            let class = classes[index];
            let trait_ref = match folded_trait_refs[class] {
                Some(trait_ref) => trait_ref,
                None => {
                    let trait_ref = node.trait_ref.fold_with(folder);
                    folded_trait_refs[class] = Some(trait_ref);
                    trait_ref
                }
            };
            let source = match node.source {
                CandidateEvidenceSource::Unique(_) => {
                    CandidateEvidenceSource::Unique(CoherenceKey { trait_ref })
                }
                source => source.fold_with(folder),
            };
            nodes.push(CandidateEvidenceNode {
                trait_ref,
                source,
                nested: node.nested,
                nested_evidence: node.nested_evidence.fold_with(folder),
            });
        }

        CandidateEvidence { root: self.root, nodes }
    }
}

impl<I: Interner> CandidateEvidence<I> {
    /// Builds a proof root from existing interned proofs. Nested recipe ordering
    /// remains semantically significant, while their handles provide sharing.
    pub fn new(
        trait_ref: ty::TraitRef<I>,
        source: CandidateEvidenceSource<I>,
        nested_evidence: impl IntoIterator<Item = CandidateEvidenceUse<I>>,
    ) -> Self {
        let nested_evidence: Vec<_> = nested_evidence.into_iter().collect();
        let evidence = CandidateEvidence {
            root: 0,
            nodes: vec![CandidateEvidenceNode {
                trait_ref,
                source,
                nested: vec![],
                nested_evidence,
            }],
        };
        evidence.assert_well_formed();
        evidence
    }

    pub fn root_node(&self) -> &CandidateEvidenceNode<I> {
        &self.nodes[usize::try_from(self.root).expect("proof node index overflow")]
    }

    pub fn root_source(&self) -> CandidateEvidenceSource<I> {
        self.root_node().source
    }

    /// Returns the concrete proof node selected by this recipe, looking
    /// through any coherence `Unique` wrapper.
    pub fn selected_node(&self) -> &CandidateEvidenceNode<I> {
        let mut index = self.root;
        for _ in 0..=self.nodes.len() {
            let node = &self.nodes[usize::try_from(index).expect("proof node index overflow")];
            match node.source {
                CandidateEvidenceSource::Unique(_) => index = node.nested[0],
                _ => return node,
            }
        }
        panic!("cycle while selecting a concrete trait proof node")
    }

    /// Returns the concrete rule selected by this recipe, looking through a
    /// coherence `Unique` wrapper when present.
    pub fn selected_source(&self) -> CandidateEvidenceSource<I> {
        self.selected_node().source
    }

    /// Wraps this concrete recipe in a coherence-quotiented `Unique` node.
    pub fn into_unique(mut self, key: CoherenceKey<I>) -> Self {
        self.assert_well_formed();
        assert_eq!(
            key.trait_ref,
            self.root_node().trait_ref,
            "coherence key does not match proof goal"
        );
        if matches!(self.root_source(), CandidateEvidenceSource::Unique(existing) if existing == key)
        {
            return self;
        }
        let selected = self.root;
        self.nodes.push(CandidateEvidenceNode {
            trait_ref: key.trait_ref,
            source: CandidateEvidenceSource::Unique(key),
            nested: vec![selected],
            nested_evidence: vec![],
        });
        self.root = u32::try_from(self.nodes.len() - 1).expect("too many proof nodes");
        self.assert_well_formed();
        self
    }

    /// Validates the node indices and acyclic shape of this proof recipe.
    pub fn assert_well_formed(&self) {
        assert!(!self.nodes.is_empty(), "empty trait proof recipe");
        let root = usize::try_from(self.root).expect("proof node index overflow");
        assert!(root < self.nodes.len(), "trait proof root is out of bounds");
        for node in &self.nodes {
            for evidence in &node.nested_evidence {
                // Nested proofs are interned. Avoid expanding shared recipes
                // into an exponential tree for structural traits with repeated fields.
                evidence.assert_well_formed();
            }
            for &nested in &node.nested {
                let nested = usize::try_from(nested).expect("proof node index overflow");
                assert!(nested < self.nodes.len(), "nested trait proof node is out of bounds");
            }
            match node.source {
                CandidateEvidenceSource::Unique(key) => {
                    assert_eq!(node.trait_ref, key.trait_ref, "invalid coherence proof key");
                    assert_eq!(node.nested.len(), 1, "unique proof must wrap one selected recipe");
                    assert!(
                        node.nested_evidence.is_empty(),
                        "unique wrapper must not own nested proofs"
                    );
                    let selected = &self.nodes
                        [usize::try_from(node.nested[0]).expect("proof node index overflow")];
                    assert_eq!(
                        node.trait_ref, selected.trait_ref,
                        "unique proof selected a recipe for a different goal"
                    );
                }
                CandidateEvidenceSource::Impl { .. }
                | CandidateEvidenceSource::Builtin { .. }
                | CandidateEvidenceSource::Dyn { .. }
                | CandidateEvidenceSource::ParamEnv { .. }
                | CandidateEvidenceSource::AliasBound(_)
                | CandidateEvidenceSource::Error
                | CandidateEvidenceSource::CoherenceUnknowable => {}
            }
        }

        fn visit<I: Interner>(index: usize, nodes: &[CandidateEvidenceNode<I>], state: &mut [u8]) {
            match state[index] {
                2 => return,
                1 => panic!("non-productive cycle in trait proof DAG"),
                0 => {}
                _ => unreachable!(),
            }
            state[index] = 1;
            for &nested in &nodes[index].nested {
                visit(usize::try_from(nested).expect("proof node index overflow"), nodes, state);
            }
            state[index] = 2;
        }

        let mut state = vec![0; self.nodes.len()];
        visit(root, &self.nodes, &mut state);
        assert!(state.into_iter().all(|state| state == 2), "unreachable node in trait proof DAG");
    }
}

/// The semantic state of a compiler-internal trait evidence value.
///
/// A selected proof recipe is only one possible value. Higher-ranked
/// instantiation may instead use a late-bound value or a universe placeholder. Error
/// evidence is explicit so recovery cannot accidentally masquerade as a
/// selected candidate.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub enum TraitEvidenceKind<I: Interner> {
    Selected(CandidateEvidence<I>),
    Bound(BoundVarIndexKind, BoundEvidence<I>),
    Placeholder(PlaceholderEvidence<I>),
    Error(I::ErrorGuaranteed),
}

impl<I: Interner> Eq for TraitEvidenceKind<I> {}

/// An interned evidence value together with the trait predicate it proves.
///
/// Keeping `trait_ref` on every state lets evidence-indexed projections remain
/// well-formed even while their proof is unresolved. For `Selected`, the field
/// is required to equal the root node's trait ref and is folded from that root
/// rather than independently, preserving shared inference identity.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub struct TraitEvidenceData<I: Interner> {
    pub trait_ref: ty::TraitRef<I>,
    pub kind: TraitEvidenceKind<I>,
}

impl<I: Interner> Eq for TraitEvidenceData<I> {}

impl<I: Interner> TraitEvidenceData<I> {
    pub fn selected(recipe: CandidateEvidence<I>) -> Self {
        recipe.assert_well_formed();
        let trait_ref = recipe.root_node().trait_ref;
        TraitEvidenceData { trait_ref, kind: TraitEvidenceKind::Selected(recipe) }
    }

    pub fn bound(
        trait_ref: ty::TraitRef<I>,
        index: BoundVarIndexKind,
        bound: BoundEvidence<I>,
    ) -> Self {
        TraitEvidenceData { trait_ref, kind: TraitEvidenceKind::Bound(index, bound) }
    }

    pub fn placeholder(trait_ref: ty::TraitRef<I>, placeholder: PlaceholderEvidence<I>) -> Self {
        TraitEvidenceData { trait_ref, kind: TraitEvidenceKind::Placeholder(placeholder) }
    }

    pub fn error(trait_ref: ty::TraitRef<I>, guar: I::ErrorGuaranteed) -> Self {
        TraitEvidenceData { trait_ref, kind: TraitEvidenceKind::Error(guar) }
    }

    pub fn as_selected(&self) -> Option<&CandidateEvidence<I>> {
        match &self.kind {
            TraitEvidenceKind::Selected(recipe) => Some(recipe),
            TraitEvidenceKind::Bound(..)
            | TraitEvidenceKind::Placeholder(_)
            | TraitEvidenceKind::Error(_) => None,
        }
    }

    pub fn into_selected(self) -> Option<CandidateEvidence<I>> {
        match self.kind {
            TraitEvidenceKind::Selected(recipe) => Some(recipe),
            TraitEvidenceKind::Bound(..)
            | TraitEvidenceKind::Placeholder(_)
            | TraitEvidenceKind::Error(_) => None,
        }
    }

    pub fn assert_well_formed(&self) {
        if let TraitEvidenceKind::Selected(recipe) = &self.kind {
            recipe.assert_well_formed();
            assert_eq!(
                self.trait_ref,
                recipe.root_node().trait_ref,
                "selected evidence predicate does not match its proof recipe"
            );
        }
    }
}

impl<I: Interner> TypeFoldable<I> for TraitEvidenceData<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        match self.kind {
            TraitEvidenceKind::Selected(recipe) => {
                Ok(TraitEvidenceData::selected(recipe.try_fold_with(folder)?))
            }
            kind => {
                Ok(TraitEvidenceData { trait_ref: self.trait_ref.try_fold_with(folder)?, kind })
            }
        }
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        match self.kind {
            TraitEvidenceKind::Selected(recipe) => {
                TraitEvidenceData::selected(recipe.fold_with(folder))
            }
            kind => TraitEvidenceData { trait_ref: self.trait_ref.fold_with(folder), kind },
        }
    }
}

impl<I: Interner> TypeVisitable<I> for TraitEvidenceData<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match &self.kind {
            // The recipe root is the authoritative occurrence of the selected
            // predicate. Visiting the duplicate field would make a shared
            // proof identity appear twice to stateful visitors.
            TraitEvidenceKind::Selected(recipe) => recipe.visit_with(visitor),
            TraitEvidenceKind::Bound(..) | TraitEvidenceKind::Placeholder(_) => {
                self.trait_ref.visit_with(visitor)
            }
            TraitEvidenceKind::Error(guar) => {
                rustc_ast_ir::try_visit!(self.trait_ref.visit_with(visitor));
                visitor.visit_error(*guar)
            }
        }
    }
}

/// One use of a nested proof recipe.
///
/// Nested evidence is interned and shares the enclosing proof's variable scope.
#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, GenericTypeVisitable)]
#[cfg_attr(
    feature = "nightly",
    derive(StableHash_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
pub enum CandidateEvidenceUse<I: Interner> {
    /// A proof in the same variable scope, folded with its parent recipe.
    Instantiated(I::TraitEvidence),
}

impl<I: Interner> Eq for CandidateEvidenceUse<I> {}

impl<I: Interner> CandidateEvidenceUse<I> {
    pub fn is_rule_only(&self) -> bool {
        let evidence = match self {
            CandidateEvidenceUse::Instantiated(evidence) => *evidence,
        };
        matches!(
            &evidence.kind,
            TraitEvidenceKind::Selected(recipe) if recipe.is_rule_only()
        )
    }

    pub fn assert_well_formed(&self) {
        // Each immutable nested recipe is validated when it is interned. Check
        // the duplicated root predicate without recursively expanding the DAG.
        let Self::Instantiated(evidence) = self;
        if let TraitEvidenceKind::Selected(recipe) = &evidence.kind {
            assert_eq!(evidence.trait_ref, recipe.root_node().trait_ref);
        }
    }
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

#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub enum ExternalRegionConstraints<I: Interner> {
    /// normal region constraints used on stable/when -Znext-solver is used by itself
    Old(Vec<(ty::RegionConstraint<I>, VisibleForLeakCheck)>),
    /// new form of region constraints used when `-Zassumptions-on-binders` is enabled.
    /// supports ORs.
    NextGen(RegionConstraint<I>),
}

impl<I: Interner> ExternalRegionConstraints<I> {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Old(r) => r.is_empty(),
            Self::NextGen(r) => r.is_true(),
        }
    }
}

/// Additional constraints returned on success.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(StableHash_NoContext))]
pub struct ExternalConstraintsData<I: Interner> {
    pub region_constraints: ExternalRegionConstraints<I>,
    pub opaque_types: Vec<(ty::OpaqueTypeKey<I>, I::Ty)>,
    pub normalization_nested_goals: NestedNormalizationGoals<I>,
}

impl<I: Interner> Eq for ExternalConstraintsData<I> {}

impl<I: Interner> ExternalConstraintsData<I> {
    pub fn new(cx: I) -> Self {
        let region_constraints = match cx.assumptions_on_binders() {
            true => ExternalRegionConstraints::NextGen(RegionConstraint::new_true()),
            false => ExternalRegionConstraints::Old(vec![]),
        };

        Self {
            region_constraints,
            opaque_types: vec![],
            normalization_nested_goals: NestedNormalizationGoals::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        let ExternalConstraintsData {
            region_constraints,
            opaque_types,
            normalization_nested_goals,
        } = self;
        region_constraints.is_empty()
            && opaque_types.is_empty()
            && normalization_nested_goals.is_empty()
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

    pub fn is_yes(&self) -> bool {
        match self {
            Certainty::Yes => true,
            Certainty::Maybe(_) => false,
        }
    }

    pub fn is_overflow(&self) -> bool {
        match self {
            Certainty::Maybe(MaybeInfo { cause: MaybeCause::Overflow { .. }, .. }) => true,
            _ => false,
        }
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

#[derive_where(Clone, Debug; I: Interner)]
pub enum SucceededInErased<I: Interner> {
    /// This goal previously succeeded in erased mode, which based on `accessed_opaques`
    /// might make us take a fast path slightly more often.
    Yes {
        accessed_opaques: AccessedOpaques<I>,
    },
    No,
}

#[derive_where(Clone, Debug; I: Interner)]
pub enum GoalStalledOnOpaques<I: Interner> {
    /// This goal got stalled in `compute_goal_fast_path`. Usually this means
    /// the goal is stalled on not that much, only one or two variables, and
    /// definitely nothing to do with opaque types. So we don't store that information.
    No,
    Yes {
        num_opaques_in_storage: usize,
        previously_succeeded_in_erased: SucceededInErased<I>,
    },
}

/// The conditions that must change for a goal to warrant
#[derive_where(Clone, Debug; I: Interner)]
pub struct GoalStalledOn<I: Interner> {
    // `ThinVec` is important for performance. See #160005.
    pub stalled_vars: ThinVec<TyOrConstInferVar>,
    // `ThinVec` is important for performance. See #160005.
    pub sub_roots: ThinVec<TyVid>,
    /// The `MaybeInfo` that will be returned on subsequent evaluations if this
    /// goal remains stalled.
    pub stalled_maybe_info: MaybeInfo,
    pub opaques: GoalStalledOnOpaques<I>,
}

/// For some goals we can trivially answer some questions without going through
/// canonicalization. There are three options:
#[derive(Clone, Debug)]
pub enum ComputeGoalFastPathOutcome<I: Interner> {
    /// Do not attempt the fast path. Compute as normal.
    NoFastPath,
    /// The goal trivially holds, immediately produce a result with [`Certainty::Yes`]
    TriviallyHolds,
    /// The goal is trivially stalled: we know for sure that it makes no sense to compute it right
    /// now, but can return information about what its stalled on and when it can be computed for real.
    TriviallyStalled { stalled_on: GoalStalledOn<I> },
}

/// Helper for `InferCtxt::ty_or_const_infer_var_changed` (see comment on that), used
/// for `traits::fulfill`'s list of `stalled_on` inference variables and for merging
/// ambiguity errors caused by the same inference variable during error reporting.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TyOrConstInferVar {
    /// Equivalent to `ty::Infer(ty::TyVar(_))`.
    Ty(TyVid),
    /// Equivalent to `ty::Infer(ty::IntVar(_))`.
    TyInt(IntVid),
    /// Equivalent to `ty::Infer(ty::FloatVar(_))`.
    TyFloat(FloatVid),

    /// Equivalent to `ty::ConstKind::Infer(ty::InferConst::Var(_))`.
    Const(ConstVid),
}

impl TyOrConstInferVar {
    pub fn as_type<I: Interner>(&self, interner: I) -> Option<I::Ty> {
        match self {
            Self::Ty(vid) => Some(I::Ty::new_var(interner, *vid)),
            Self::TyInt(_) | Self::TyFloat(_) | Self::Const(_) => None,
        }
    }

    /// Tries to extract an inference variable from a type or a constant, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`) and
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    pub fn maybe_from_generic_arg<I: Interner>(arg: I::GenericArg) -> Option<Self> {
        match arg.kind() {
            GenericArgKind::Type(ty) => Self::maybe_from_ty::<I>(ty),
            GenericArgKind::Const(ct) => Self::maybe_from_const::<I>(ct),
            GenericArgKind::Lifetime(_) => None,
        }
    }

    /// Tries to extract an inference variable from a type or a constant, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`) and
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    pub fn maybe_from_term<I: Interner>(term: I::Term) -> Option<Self> {
        match term.kind() {
            TermKind::Ty(ty) => Self::maybe_from_ty::<I>(ty),
            TermKind::Const(ct) => Self::maybe_from_const::<I>(ct),
        }
    }

    /// Tries to extract an inference variable from a type, returns `None`
    /// for types other than `ty::Infer(_)` (or `InferTy::Fresh*`).
    fn maybe_from_ty<I: Interner>(ty: I::Ty) -> Option<Self> {
        match ty.kind() {
            ty::Infer(ty::TyVar(v)) => Some(TyOrConstInferVar::Ty(v)),
            ty::Infer(ty::IntVar(v)) => Some(TyOrConstInferVar::TyInt(v)),
            ty::Infer(ty::FloatVar(v)) => Some(TyOrConstInferVar::TyFloat(v)),
            _ => None,
        }
    }

    /// Tries to extract an inference variable from a constant, returns `None`
    /// for constants other than `ty::ConstKind::Infer(_)` (or `InferConst::Fresh`).
    fn maybe_from_const<I: Interner>(ct: I::Const) -> Option<Self> {
        match ct.kind() {
            ty::ConstKind::Infer(InferConst::Var(v)) => Some(TyOrConstInferVar::Const(v)),
            _ => None,
        }
    }
}
