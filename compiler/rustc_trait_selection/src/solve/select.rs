use std::ops::ControlFlow;

use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::solve::inspect::ProbeKind;
use rustc_infer::traits::solve::{CandidateSource, Certainty, Goal};
use rustc_infer::traits::{
    BuiltinImplSource, ImplSource, ImplSourceUserDefinedData, Obligation, ObligationCause,
    Selection, SelectionError, SelectionResult, TraitObligation,
};
use rustc_macros::extension;
use rustc_middle::{bug, span_bug};
use rustc_span::Span;

use crate::solve::inspect::{self, ProofTreeInferCtxtExt};

#[extension(pub trait InferCtxtSelectExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    fn select_in_new_trait_solver(
        &self,
        obligation: &TraitObligation<'tcx>,
    ) -> SelectionResult<'tcx, Selection<'tcx>> {
        assert!(self.next_trait_solver());

        self.visit_proof_tree(
            Goal::new(self.tcx, obligation.param_env, obligation.predicate),
            &mut Select { span: obligation.cause.span },
        )
        .break_value()
        .unwrap()
    }
}

struct Select {
    span: Span,
}

impl<'tcx> inspect::ProofTreeVisitor<'tcx> for Select {
    type Result = ControlFlow<SelectionResult<'tcx, Selection<'tcx>>>;

    fn span(&self) -> Span {
        self.span
    }

    fn visit_goal(&mut self, goal: &inspect::InspectGoal<'_, 'tcx>) -> Self::Result {
        let mut candidates = goal.candidates();
        candidates.retain(|cand| cand.result().is_ok());

        // No candidates -- not implemented.
        if candidates.is_empty() {
            return ControlFlow::Break(Err(SelectionError::Unimplemented));
        }

        // One candidate, no need to winnow.
        if candidates.len() == 1 {
            return ControlFlow::Break(Ok(to_selection(
                self.span,
                candidates.into_iter().next().unwrap(),
            )));
        }

        // Don't winnow until `Certainty::Yes` -- we don't need to winnow until
        // codegen, and only on the good path.
        if matches!(goal.result().unwrap(), Certainty::Maybe(..)) {
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

        ControlFlow::Break(Ok(to_selection(self.span, candidates.into_iter().next().unwrap())))
    }
}

/// This is a lot more limited than the old solver's equivalent method. This may lead to more `Ok(None)`
/// results when selecting traits in polymorphic contexts, but we should never rely on the lack of ambiguity,
/// and should always just gracefully fail here. We shouldn't rely on this incompleteness.
fn candidate_should_be_dropped_in_favor_of<'tcx>(
    victim: &inspect::InspectCandidate<'_, 'tcx>,
    other: &inspect::InspectCandidate<'_, 'tcx>,
) -> bool {
    // Don't winnow until `Certainty::Yes` -- we don't need to winnow until
    // codegen, and only on the good path.
    if matches!(other.result().unwrap(), Certainty::Maybe(..)) {
        return false;
    }

    let inspect::ProbeKind::TraitCandidate { source: victim_source, result: _ } = victim.kind()
    else {
        return false;
    };
    let inspect::ProbeKind::TraitCandidate { source: other_source, result: _ } = other.kind()
    else {
        return false;
    };

    match (victim_source, other_source) {
        (_, CandidateSource::CoherenceUnknowable) | (CandidateSource::CoherenceUnknowable, _) => {
            bug!("should not have assembled a CoherenceUnknowable candidate")
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
            CandidateSource::Impl(_) | CandidateSource::ParamEnv(_) | CandidateSource::AliasBound,
            CandidateSource::BuiltinImpl(BuiltinImplSource::Object { .. }),
        ) => true,

        // Prefer specializing candidates over specialized candidates.
        (CandidateSource::Impl(victim_def_id), CandidateSource::Impl(other_def_id)) => {
            victim.goal().infcx().tcx.specializes((other_def_id, victim_def_id))
        }

        _ => false,
    }
}

fn to_selection<'tcx>(
    span: Span,
    cand: inspect::InspectCandidate<'_, 'tcx>,
) -> Option<Selection<'tcx>> {
    if let Certainty::Maybe(..) = cand.shallow_certainty() {
        return None;
    }

    let (nested, impl_args) = cand.instantiate_nested_goals_and_opt_impl_args(span);
    let nested = nested
        .into_iter()
        .map(|nested| {
            Obligation::new(
                nested.infcx().tcx,
                ObligationCause::dummy_with_span(span),
                nested.goal().param_env,
                nested.goal().predicate,
            )
        })
        .collect();

    Some(match cand.kind() {
        ProbeKind::TraitCandidate { source, result: _ } => match source {
            CandidateSource::Impl(impl_def_id) => {
                // FIXME: Remove this in favor of storing this in the tree
                // For impl candidates, we do the rematch manually to compute the args.
                ImplSource::UserDefined(ImplSourceUserDefinedData {
                    impl_def_id,
                    args: impl_args.expect("expected recorded impl args for impl candidate"),
                    nested,
                })
            }
            CandidateSource::BuiltinImpl(builtin) => ImplSource::Builtin(builtin, nested),
            CandidateSource::ParamEnv(_) | CandidateSource::AliasBound => ImplSource::Param(nested),
            CandidateSource::CoherenceUnknowable => {
                span_bug!(span, "didn't expect to select an unknowable candidate")
            }
        },
        ProbeKind::NormalizedSelfTyAssembly
        | ProbeKind::UnsizeAssembly
        | ProbeKind::ProjectionCompatibility
        | ProbeKind::OpaqueTypeStorageLookup { result: _ }
        | ProbeKind::Root { result: _ }
        | ProbeKind::ShadowedEnvProbing
        | ProbeKind::RigidAlias { result: _ } => {
            span_bug!(span, "didn't expect to assemble trait candidate from {:#?}", cand.kind())
        }
    })
}
