#![expect(dead_code, reason = "this is used by rustc")]

use std::ops::ControlFlow;

use hir_def::{ImplId, TraitId};
use macros::{TypeFoldable, TypeVisitable};
use rustc_type_ir::{
    Interner,
    solve::{BuiltinImplSource, CandidateSource, Certainty, inspect::ProbeKind},
};

use crate::{
    db::InternedOpaqueTyId,
    next_solver::{
        Const, ErrorGuaranteed, GenericArgs, Goal, TraitRef, Ty, TypeError,
        infer::{
            InferCtxt,
            select::EvaluationResult::*,
            traits::{Obligation, ObligationCause, PredicateObligation, TraitObligation},
        },
        inspect::{InspectCandidate, InspectGoal, ProofTreeVisitor},
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError<'db> {
    /// The trait is not implemented.
    Unimplemented,
    /// After a closure impl has selected, its "outputs" were evaluated
    /// (which for closures includes the "input" type params) and they
    /// didn't resolve. See `confirm_poly_trait_refs` for more.
    SignatureMismatch(Box<SignatureMismatchData<'db>>),
    /// The trait pointed by `DefId` is dyn-incompatible.
    TraitDynIncompatible(TraitId),
    /// A given constant couldn't be evaluated.
    NotConstEvaluatable(NotConstEvaluatable),
    /// Exceeded the recursion depth during type projection.
    Overflow(OverflowError),
    /// Computing an opaque type's hidden type caused an error (e.g. a cycle error).
    /// We can thus not know whether the hidden type implements an auto trait, so
    /// we should not presume anything about it.
    OpaqueTypeAutoTraitLeakageUnknown(InternedOpaqueTyId),
    /// Error for a `ConstArgHasType` goal
    ConstArgHasWrongType { ct: Const<'db>, ct_ty: Ty<'db>, expected_ty: Ty<'db> },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum NotConstEvaluatable {
    Error(ErrorGuaranteed),
    MentionsInfer,
    MentionsParam,
}

/// The result of trait evaluation. The order is important
/// here as the evaluation of a list is the maximum of the
/// evaluations.
///
/// The evaluation results are ordered:
///     - `EvaluatedToOk` implies `EvaluatedToOkModuloRegions`
///       implies `EvaluatedToAmbig` implies `EvaluatedToAmbigStackDependent`
///     - the "union" of evaluation results is equal to their maximum -
///     all the "potential success" candidates can potentially succeed,
///     so they are noops when unioned with a definite error, and within
///     the categories it's easy to see that the unions are correct.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum EvaluationResult {
    /// Evaluation successful.
    EvaluatedToOk,
    /// Evaluation successful, but there were unevaluated region obligations.
    EvaluatedToOkModuloRegions,
    /// Evaluation successful, but need to rerun because opaque types got
    /// hidden types assigned without it being known whether the opaque types
    /// are within their defining scope
    EvaluatedToOkModuloOpaqueTypes,
    /// Evaluation is known to be ambiguous -- it *might* hold for some
    /// assignment of inference variables, but it might not.
    ///
    /// While this has the same meaning as `EvaluatedToAmbigStackDependent` -- we can't
    /// know whether this obligation holds or not -- it is the result we
    /// would get with an empty stack, and therefore is cacheable.
    EvaluatedToAmbig,
    /// Evaluation failed because of recursion involving inference
    /// variables. We are somewhat imprecise there, so we don't actually
    /// know the real result.
    ///
    /// This can't be trivially cached because the result depends on the
    /// stack results.
    EvaluatedToAmbigStackDependent,
    /// Evaluation failed.
    EvaluatedToErr,
}

impl EvaluationResult {
    /// Returns `true` if this evaluation result is known to apply, even
    /// considering outlives constraints.
    pub(crate) fn must_apply_considering_regions(self) -> bool {
        self == EvaluatedToOk
    }

    /// Returns `true` if this evaluation result is known to apply, ignoring
    /// outlives constraints.
    pub(crate) fn must_apply_modulo_regions(self) -> bool {
        self <= EvaluatedToOkModuloRegions
    }

    pub(crate) fn may_apply(self) -> bool {
        match self {
            EvaluatedToOkModuloOpaqueTypes
            | EvaluatedToOk
            | EvaluatedToOkModuloRegions
            | EvaluatedToAmbig
            | EvaluatedToAmbigStackDependent => true,

            EvaluatedToErr => false,
        }
    }

    pub(crate) fn is_stack_dependent(self) -> bool {
        match self {
            EvaluatedToAmbigStackDependent => true,

            EvaluatedToOkModuloOpaqueTypes
            | EvaluatedToOk
            | EvaluatedToOkModuloRegions
            | EvaluatedToAmbig
            | EvaluatedToErr => false,
        }
    }
}

/// Indicates that trait evaluation caused overflow and in which pass.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum OverflowError {
    Error(ErrorGuaranteed),
    Canonical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignatureMismatchData<'db> {
    pub(crate) found_trait_ref: TraitRef<'db>,
    pub(crate) expected_trait_ref: TraitRef<'db>,
    pub(crate) terr: TypeError<'db>,
}

/// When performing resolution, it is typically the case that there
/// can be one of three outcomes:
///
/// - `Ok(Some(r))`: success occurred with result `r`
/// - `Ok(None)`: could not definitely determine anything, usually due
///   to inconclusive type inference.
/// - `Err(e)`: error `e` occurred
pub(crate) type SelectionResult<'db, T> = Result<Option<T>, SelectionError<'db>>;

/// Given the successful resolution of an obligation, the `ImplSource`
/// indicates where the impl comes from.
///
/// For example, the obligation may be satisfied by a specific impl (case A),
/// or it may be relative to some bound that is in scope (case B).
///
/// ```ignore (illustrative)
/// impl<T:Clone> Clone<T> for Option<T> { ... } // Impl_1
/// impl<T:Clone> Clone<T> for Box<T> { ... }    // Impl_2
/// impl Clone for i32 { ... }                   // Impl_3
///
/// fn foo<T: Clone>(concrete: Option<Box<i32>>, param: T, mixed: Option<T>) {
///     // Case A: ImplSource points at a specific impl. Only possible when
///     // type is concretely known. If the impl itself has bounded
///     // type parameters, ImplSource will carry resolutions for those as well:
///     concrete.clone(); // ImplSource(Impl_1, [ImplSource(Impl_2, [ImplSource(Impl_3)])])
///
///     // Case B: ImplSource must be provided by caller. This applies when
///     // type is a type parameter.
///     param.clone();    // ImplSource::Param
///
///     // Case C: A mix of cases A and B.
///     mixed.clone();    // ImplSource(Impl_1, [ImplSource::Param])
/// }
/// ```
///
/// ### The type parameter `N`
///
/// See explanation on `ImplSourceUserDefinedData`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeVisitable, TypeFoldable)]
pub(crate) enum ImplSource<'db, N> {
    /// ImplSource identifying a particular impl.
    UserDefined(ImplSourceUserDefinedData<'db, N>),

    /// Successful resolution to an obligation provided by the caller
    /// for some type parameter. The `Vec<N>` represents the
    /// obligations incurred from normalizing the where-clause (if
    /// any).
    Param(Vec<N>),

    /// Successful resolution for a builtin impl.
    Builtin(BuiltinImplSource, Vec<N>),
}

impl<'db, N> ImplSource<'db, N> {
    pub(crate) fn nested_obligations(self) -> Vec<N> {
        match self {
            ImplSource::UserDefined(i) => i.nested,
            ImplSource::Param(n) | ImplSource::Builtin(_, n) => n,
        }
    }

    pub(crate) fn borrow_nested_obligations(&self) -> &[N] {
        match self {
            ImplSource::UserDefined(i) => &i.nested,
            ImplSource::Param(n) | ImplSource::Builtin(_, n) => n,
        }
    }

    pub(crate) fn borrow_nested_obligations_mut(&mut self) -> &mut [N] {
        match self {
            ImplSource::UserDefined(i) => &mut i.nested,
            ImplSource::Param(n) | ImplSource::Builtin(_, n) => n,
        }
    }

    pub(crate) fn map<M, F>(self, f: F) -> ImplSource<'db, M>
    where
        F: FnMut(N) -> M,
    {
        match self {
            ImplSource::UserDefined(i) => ImplSource::UserDefined(ImplSourceUserDefinedData {
                impl_def_id: i.impl_def_id,
                args: i.args,
                nested: i.nested.into_iter().map(f).collect(),
            }),
            ImplSource::Param(n) => ImplSource::Param(n.into_iter().map(f).collect()),
            ImplSource::Builtin(source, n) => {
                ImplSource::Builtin(source, n.into_iter().map(f).collect())
            }
        }
    }
}

/// Identifies a particular impl in the source, along with a set of
/// generic parameters from the impl's type/lifetime parameters. The
/// `nested` vector corresponds to the nested obligations attached to
/// the impl's type parameters.
///
/// The type parameter `N` indicates the type used for "nested
/// obligations" that are required by the impl. During type-check, this
/// is `Obligation`, as one might expect. During codegen, however, this
/// is `()`, because codegen only requires a shallow resolution of an
/// impl, and nested obligations are satisfied later.
#[derive(Debug, Clone, PartialEq, Eq, Hash, TypeVisitable, TypeFoldable)]
pub(crate) struct ImplSourceUserDefinedData<'db, N> {
    #[type_visitable(ignore)]
    #[type_foldable(identity)]
    pub(crate) impl_def_id: ImplId,
    pub(crate) args: GenericArgs<'db>,
    pub(crate) nested: Vec<N>,
}

pub(crate) type Selection<'db> = ImplSource<'db, PredicateObligation<'db>>;

impl<'db> InferCtxt<'db> {
    pub(crate) fn select(
        &self,
        obligation: &TraitObligation<'db>,
    ) -> SelectionResult<'db, Selection<'db>> {
        self.visit_proof_tree(
            Goal::new(self.interner, obligation.param_env, obligation.predicate),
            &mut Select {},
        )
        .break_value()
        .unwrap()
    }
}

struct Select {}

impl<'db> ProofTreeVisitor<'db> for Select {
    type Result = ControlFlow<SelectionResult<'db, Selection<'db>>>;

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'db>) -> Self::Result {
        let mut candidates = goal.candidates();
        candidates.retain(|cand| cand.result().is_ok());

        // No candidates -- not implemented.
        if candidates.is_empty() {
            return ControlFlow::Break(Err(SelectionError::Unimplemented));
        }

        // One candidate, no need to winnow.
        if candidates.len() == 1 {
            return ControlFlow::Break(Ok(to_selection(candidates.into_iter().next().unwrap())));
        }

        // Don't winnow until `Certainty::Yes` -- we don't need to winnow until
        // codegen, and only on the good path.
        if matches!(goal.result().unwrap(), Certainty::Maybe { .. }) {
            return ControlFlow::Break(Ok(None));
        }

        // We need to winnow. See comments on `candidate_should_be_dropped_in_favor_of`.
        let mut i = 0;
        while i < candidates.len() {
            let should_drop_i = (0..candidates.len())
                .filter(|&j| i != j)
                .any(|j| candidate_should_be_dropped_in_favor_of(&candidates[i], &candidates[j]));
            if should_drop_i {
                candidates.swap_remove(i);
            } else {
                i += 1;
                if i > 1 {
                    return ControlFlow::Break(Ok(None));
                }
            }
        }

        ControlFlow::Break(Ok(to_selection(candidates.into_iter().next().unwrap())))
    }
}

/// This is a lot more limited than the old solver's equivalent method. This may lead to more `Ok(None)`
/// results when selecting traits in polymorphic contexts, but we should never rely on the lack of ambiguity,
/// and should always just gracefully fail here. We shouldn't rely on this incompleteness.
fn candidate_should_be_dropped_in_favor_of<'db>(
    victim: &InspectCandidate<'_, 'db>,
    other: &InspectCandidate<'_, 'db>,
) -> bool {
    // Don't winnow until `Certainty::Yes` -- we don't need to winnow until
    // codegen, and only on the good path.
    if matches!(other.result().unwrap(), Certainty::Maybe { .. }) {
        return false;
    }

    let ProbeKind::TraitCandidate { source: victim_source, result: _ } = victim.kind() else {
        return false;
    };
    let ProbeKind::TraitCandidate { source: other_source, result: _ } = other.kind() else {
        return false;
    };

    match (victim_source, other_source) {
        (_, CandidateSource::CoherenceUnknowable) | (CandidateSource::CoherenceUnknowable, _) => {
            panic!("should not have assembled a CoherenceUnknowable candidate")
        }

        // In the old trait solver, we arbitrarily choose lower vtable candidates
        // over higher ones.
        (
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object(a)),
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object(b)),
        ) => a >= b,
        (
            CandidateSource::BuiltinImpl(BuiltinImplSource::TraitUpcasting(a)),
            CandidateSource::BuiltinImpl(BuiltinImplSource::TraitUpcasting(b)),
        ) => a >= b,
        // Prefer dyn candidates over non-dyn candidates. This is necessary to
        // handle the unsoundness between `impl<T: ?Sized> Any for T` and `dyn Any: Any`.
        (
            CandidateSource::Impl(_)
            | CandidateSource::ParamEnv(_)
            | CandidateSource::AliasBound(_),
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object { .. }),
        ) => true,

        // Prefer specializing candidates over specialized candidates.
        (CandidateSource::Impl(victim_def_id), CandidateSource::Impl(other_def_id)) => {
            victim.goal().infcx().interner.impl_specializes(other_def_id, victim_def_id)
        }

        _ => false,
    }
}

fn to_selection<'db>(cand: InspectCandidate<'_, 'db>) -> Option<Selection<'db>> {
    if let Certainty::Maybe { .. } = cand.shallow_certainty() {
        return None;
    }

    let nested = match cand.result().expect("expected positive result") {
        Certainty::Yes => Vec::new(),
        Certainty::Maybe { .. } => cand
            .instantiate_nested_goals()
            .into_iter()
            .map(|nested| {
                Obligation::new(
                    nested.infcx().interner,
                    ObligationCause::dummy(),
                    nested.goal().param_env,
                    nested.goal().predicate,
                )
            })
            .collect(),
    };

    Some(match cand.kind() {
        ProbeKind::TraitCandidate { source, result: _ } => match source {
            CandidateSource::Impl(impl_def_id) => {
                // FIXME: Remove this in favor of storing this in the tree
                // For impl candidates, we do the rematch manually to compute the args.
                ImplSource::UserDefined(ImplSourceUserDefinedData {
                    impl_def_id: impl_def_id.0,
                    args: cand.instantiate_impl_args(),
                    nested,
                })
            }
            CandidateSource::BuiltinImpl(builtin) => ImplSource::Builtin(builtin, nested),
            CandidateSource::ParamEnv(_) | CandidateSource::AliasBound(_) => {
                ImplSource::Param(nested)
            }
            CandidateSource::CoherenceUnknowable => {
                panic!("didn't expect to select an unknowable candidate")
            }
        },
        ProbeKind::NormalizedSelfTyAssembly
        | ProbeKind::UnsizeAssembly
        | ProbeKind::ProjectionCompatibility
        | ProbeKind::OpaqueTypeStorageLookup { result: _ }
        | ProbeKind::Root { result: _ }
        | ProbeKind::ShadowedEnvProbing
        | ProbeKind::RigidAlias { result: _ } => {
            panic!("didn't expect to assemble trait candidate from {:#?}", cand.kind())
        }
    })
}
