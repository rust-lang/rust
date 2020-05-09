//! Defines a Chalk-based `TraitEngine`

use crate::infer::canonical::OriginalQueryValues;
use crate::infer::InferCtxt;
use crate::traits::query::NoSolution;
use crate::traits::{
    ChalkEnvironmentAndGoal, ChalkEnvironmentClause, FulfillmentError, FulfillmentErrorCode,
    ObligationCause, PredicateObligation, SelectionError, TraitEngine,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct FulfillmentContext<'tcx> {
    obligations: FxHashSet<PredicateObligation<'tcx>>,
}

impl FulfillmentContext<'tcx> {
    crate fn new() -> Self {
        FulfillmentContext { obligations: FxHashSet::default() }
    }
}

fn environment<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
) -> &'tcx ty::List<ChalkEnvironmentClause<'tcx>> {
    use rustc_hir::{ForeignItemKind, ImplItemKind, ItemKind, Node, TraitItemKind};
    use rustc_middle::ty::subst::GenericArgKind;

    debug!("environment(def_id = {:?})", def_id);

    // The environment of an impl Trait type is its defining function's environment.
    if let Some(parent) = ty::is_impl_trait_defn(tcx, def_id) {
        return environment(tcx, parent);
    }

    // Compute the bounds on `Self` and the type parameters.
    let ty::InstantiatedPredicates { predicates, .. } =
        tcx.predicates_of(def_id).instantiate_identity(tcx);

    let clauses = predicates.into_iter().map(|pred| ChalkEnvironmentClause::Predicate(pred));

    let hir_id = tcx.hir().as_local_hir_id(def_id.expect_local());
    let node = tcx.hir().get(hir_id);

    enum NodeKind {
        TraitImpl,
        InherentImpl,
        Fn,
        Other,
    };

    let node_kind = match node {
        Node::TraitItem(item) => match item.kind {
            TraitItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        Node::ImplItem(item) => match item.kind {
            ImplItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        Node::Item(item) => match item.kind {
            ItemKind::Impl { of_trait: Some(_), .. } => NodeKind::TraitImpl,
            ItemKind::Impl { of_trait: None, .. } => NodeKind::InherentImpl,
            ItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        Node::ForeignItem(item) => match item.kind {
            ForeignItemKind::Fn(..) => NodeKind::Fn,
            _ => NodeKind::Other,
        },

        // FIXME: closures?
        _ => NodeKind::Other,
    };

    // FIXME(eddyb) isn't the unordered nature of this a hazard?
    let mut inputs = FxHashSet::default();

    match node_kind {
        // In a trait impl, we assume that the header trait ref and all its
        // constituents are well-formed.
        NodeKind::TraitImpl => {
            let trait_ref = tcx.impl_trait_ref(def_id).expect("not an impl");

            inputs.extend(trait_ref.substs.iter().flat_map(|&arg| arg.walk()));
        }

        // In an inherent impl, we assume that the receiver type and all its
        // constituents are well-formed.
        NodeKind::InherentImpl => {
            let self_ty = tcx.type_of(def_id);
            inputs.extend(self_ty.walk());
        }

        // In an fn, we assume that the arguments and all their constituents are
        // well-formed.
        NodeKind::Fn => {
            let fn_sig = tcx.fn_sig(def_id);
            let fn_sig = tcx.liberate_late_bound_regions(def_id, &fn_sig);

            inputs.extend(fn_sig.inputs().iter().flat_map(|ty| ty.walk()));
        }

        NodeKind::Other => (),
    }
    let input_clauses = inputs.into_iter().filter_map(|arg| {
        match arg.unpack() {
            GenericArgKind::Type(ty) => Some(ChalkEnvironmentClause::TypeFromEnv(ty)),

            // FIXME(eddyb) no WF conditions from lifetimes?
            GenericArgKind::Lifetime(_) => None,

            // FIXME(eddyb) support const generics in Chalk
            GenericArgKind::Const(_) => None,
        }
    });

    tcx.mk_chalk_environment_clause_list(clauses.chain(input_clauses))
}

/// We need to wrap a `ty::Predicate` in an elaborated environment *before* we
/// canonicalize. This is due to the fact that we insert extra clauses into the
/// environment for all input types (`FromEnv`).
fn in_environment(
    infcx: &InferCtxt<'_, 'tcx>,
    obligation: &PredicateObligation<'tcx>,
) -> ChalkEnvironmentAndGoal<'tcx> {
    assert!(!infcx.is_in_snapshot());
    let obligation = infcx.resolve_vars_if_possible(obligation);

    let environment = match obligation.param_env.def_id {
        Some(def_id) => environment(infcx.tcx, def_id),
        None if obligation.param_env.caller_bounds.is_empty() => ty::List::empty(),
        _ => bug!("non-empty `ParamEnv` with no def-id"),
    };

    ChalkEnvironmentAndGoal { environment, goal: obligation.predicate }
}

impl TraitEngine<'tcx> for FulfillmentContext<'tcx> {
    fn normalize_projection_type(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        _param_env: ty::ParamEnv<'tcx>,
        projection_ty: ty::ProjectionTy<'tcx>,
        _cause: ObligationCause<'tcx>,
    ) -> Ty<'tcx> {
        infcx.tcx.mk_ty(ty::Projection(projection_ty))
    }

    fn register_predicate_obligation(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
        obligation: PredicateObligation<'tcx>,
    ) {
        assert!(!infcx.is_in_snapshot());
        let obligation = infcx.resolve_vars_if_possible(&obligation);

        self.obligations.insert(obligation);
    }

    fn select_all_or_error(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        self.select_where_possible(infcx)?;

        if self.obligations.is_empty() {
            Ok(())
        } else {
            let errors = self
                .obligations
                .iter()
                .map(|obligation| FulfillmentError {
                    obligation: obligation.clone(),
                    code: FulfillmentErrorCode::CodeAmbiguity,
                    points_at_arg_span: false,
                })
                .collect();
            Err(errors)
        }
    }

    fn select_where_possible(
        &mut self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Result<(), Vec<FulfillmentError<'tcx>>> {
        let mut errors = Vec::new();
        let mut next_round = FxHashSet::default();
        let mut making_progress;

        loop {
            making_progress = false;

            // We iterate over all obligations, and record if we are able
            // to unambiguously prove at least one obligation.
            for obligation in self.obligations.drain() {
                let goal_in_environment = in_environment(infcx, &obligation);
                let mut orig_values = OriginalQueryValues::default();
                let canonical_goal =
                    infcx.canonicalize_query(&goal_in_environment, &mut orig_values);

                match infcx.tcx.evaluate_goal(canonical_goal) {
                    Ok(response) => {
                        if response.is_proven() {
                            making_progress = true;

                            match infcx.instantiate_query_response_and_region_obligations(
                                &obligation.cause,
                                obligation.param_env,
                                &orig_values,
                                &response,
                            ) {
                                Ok(infer_ok) => next_round.extend(
                                    infer_ok.obligations.into_iter().map(|obligation| {
                                        assert!(!infcx.is_in_snapshot());
                                        infcx.resolve_vars_if_possible(&obligation)
                                    }),
                                ),

                                Err(_err) => errors.push(FulfillmentError {
                                    obligation: obligation,
                                    code: FulfillmentErrorCode::CodeSelectionError(
                                        SelectionError::Unimplemented,
                                    ),
                                    points_at_arg_span: false,
                                }),
                            }
                        } else {
                            // Ambiguous: retry at next round.
                            next_round.insert(obligation);
                        }
                    }

                    Err(NoSolution) => errors.push(FulfillmentError {
                        obligation: obligation,
                        code: FulfillmentErrorCode::CodeSelectionError(
                            SelectionError::Unimplemented,
                        ),
                        points_at_arg_span: false,
                    }),
                }
            }
            next_round = std::mem::replace(&mut self.obligations, next_round);

            if !making_progress {
                break;
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn pending_obligations(&self) -> Vec<PredicateObligation<'tcx>> {
        self.obligations.iter().map(|obligation| obligation.clone()).collect()
    }
}
