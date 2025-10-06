pub mod inspect;

use std::hash::Hash;

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::lang_items::SolverTraitLangItem;
use crate::search_graph::PathKind;
use crate::{self as ty, Canonical, CanonicalVarValues, Interner, Upcast};

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
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct NoSolution;

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, P)]
#[derive_where(Copy; I: Interner, P: Copy)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
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
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
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
    /// Instantiating a higher-ranked goal and re-proving it.
    InstantiateHigherRanked,
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
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Decodable_NoContext, Encodable_NoContext, HashStable_NoContext)
)]
pub struct QueryInput<I: Interner, P> {
    pub goal: Goal<I, P>,
    pub predefined_opaques_in_body: I::PredefinedOpaques,
}

impl<I: Interner, P: Eq> Eq for QueryInput<I, P> {}

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
    AliasBound,
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
#[cfg_attr(
    feature = "nightly",
    derive(HashStable_NoContext, Encodable_NoContext, Decodable_NoContext)
)]
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

#[derive_where(Clone, Copy, Hash, PartialEq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct Response<I: Interner> {
    pub certainty: Certainty,
    pub var_values: CanonicalVarValues<I>,
    /// Additional constraints returned by this query.
    pub external_constraints: I::ExternalConstraints,
}

impl<I: Interner> Eq for Response<I> {}

/// Additional constraints returned on success.
#[derive_where(Clone, Hash, PartialEq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct ExternalConstraintsData<I: Interner> {
    pub region_constraints: Vec<ty::OutlivesPredicate<I, I::GenericArg>>,
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

#[derive_where(Clone, Hash, PartialEq, Debug, Default; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
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
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum Certainty {
    Yes,
    Maybe { cause: MaybeCause, opaque_types_jank: OpaqueTypesJank },
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
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
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

impl Certainty {
    pub const AMBIGUOUS: Certainty = Certainty::Maybe {
        cause: MaybeCause::Ambiguity,
        opaque_types_jank: OpaqueTypesJank::AllGood,
    };

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
            (
                Certainty::Maybe { cause: a_cause, opaque_types_jank: a_jank },
                Certainty::Maybe { cause: b_cause, opaque_types_jank: b_jank },
            ) => Certainty::Maybe {
                cause: a_cause.and(b_cause),
                opaque_types_jank: a_jank.and(b_jank),
            },
        }
    }

    pub const fn overflow(suggest_increasing_limit: bool) -> Certainty {
        Certainty::Maybe {
            cause: MaybeCause::Overflow { suggest_increasing_limit, keep_constraints: false },
            opaque_types_jank: OpaqueTypesJank::AllGood,
        }
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
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
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
