use std::ops::ControlFlow;

use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_middle::ty;
use rustc_span::def_id::DefId;
use rustc_trait_selection::solve::CandidateSource;
use rustc_trait_selection::solve::inspect::{
    InferCtxtProofTreeExt as _, InspectGoal, ProbeKind, ProofTreeVisitor,
};

use crate::clean;
use crate::clean::{
    clean_middle_assoc_item, clean_middle_ty, clean_trait_ref_with_constraints, clean_ty_generics,
};
use crate::core::DocContext;

#[tracing::instrument(level = "debug", skip(cx))]
pub(crate) fn synthesize_blanket_impls(
    cx: &mut DocContext<'_>,
    item_def_id: DefId,
) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let item_ty = tcx.type_of(item_def_id);
    let param_env = ty::ParamEnv::empty();
    let typing_mode = ty::TypingMode::non_body_analysis();

    let mut blanket_impls = Vec::new();

    for trait_def_id in tcx.visible_traits() {
        if !cx.cache.effective_visibilities.is_reachable(tcx, trait_def_id)
            || cx.generated_synthetics.contains(&(item_ty.skip_binder(), trait_def_id))
        {
            continue;
        }

        // FIXME(fmease): ...
        if tcx.is_lang_item(trait_def_id, hir::LangItem::PointeeSized) {
            continue;
        }

        let infcx = tcx.infer_ctxt().with_next_trait_solver(true).build(typing_mode);

        let item_args = infcx.fresh_args_for_item(rustc_span::DUMMY_SP, item_def_id);
        let self_ty = item_ty.instantiate(tcx, item_args).skip_normalization();
        let trait_args = infcx.fresh_args_for_item(rustc_span::DUMMY_SP, trait_def_id);
        let trait_ref = ty::TraitRef::new_from_args(tcx, trait_def_id, trait_args)
            .with_replaced_self_ty(tcx, self_ty);
        let goal = ty::solve::Goal::new(tcx, param_env, trait_ref);

        let ControlFlow::Break(impl_def_id) = infcx.visit_proof_tree(goal, &mut HasBlanketImpl)
        else {
            continue;
        };

        cx.generated_synthetics.insert((item_ty.skip_binder(), trait_def_id));

        let item_ty = item_ty.instantiate_identity().skip_normalization();
        let impl_trait_ref =
            tcx.impl_trait_ref(impl_def_id).instantiate_identity().skip_normalization();

        blanket_impls.push(clean::Item {
            inner: Box::new(clean::ItemInner {
                name: None,
                item_id: clean::ItemId::Blanket { impl_id: impl_def_id, for_: item_def_id },
                attrs: Default::default(),
                stability: None,
                kind: clean::ImplItem(Box::new(clean::Impl {
                    safety: hir::Safety::Safe,
                    generics: clean_ty_generics(cx, impl_def_id),
                    // FIXME(eddyb) compute both `trait_` and `for_` from
                    // the post-inference `trait_ref`, as it's more accurate.
                    trait_: Some(clean_trait_ref_with_constraints(
                        cx,
                        ty::Binder::dummy(impl_trait_ref),
                        ThinVec::new(),
                    )),
                    for_: clean_middle_ty(ty::Binder::dummy(item_ty), cx, None, None),
                    items: tcx
                        .associated_items(impl_def_id)
                        .in_definition_order()
                        .filter(|item| !item.is_impl_trait_in_trait())
                        .map(|item| clean_middle_assoc_item(item, cx))
                        .collect(),
                    polarity: ty::ImplPolarity::Positive,
                    kind: clean::ImplKind::Blanket(Box::new(clean_middle_ty(
                        ty::Binder::dummy(impl_trait_ref.self_ty()),
                        cx,
                        None,
                        None,
                    ))),
                    is_deprecated: tcx
                        .lookup_deprecation(impl_def_id)
                        .is_some_and(|deprecation| deprecation.is_in_effect()),
                })),
                cfg: None,
                inline_stmt_id: None,
            }),
        });
    }

    blanket_impls
}

struct HasBlanketImpl;

impl<'tcx> ProofTreeVisitor<'tcx> for HasBlanketImpl {
    type Result = ControlFlow<DefId>;

    fn span(&self) -> rustc_span::Span {
        rustc_span::DUMMY_SP
    }

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> Self::Result {
        for candidate in goal.candidates() {
            if candidate.result().is_ok()
                && let ProbeKind::TraitCandidate { source, .. } = candidate.kind()
                && let CandidateSource::Impl(impl_def_id) = source
                && let ty::Param(_) = goal.infcx().tcx.type_of(impl_def_id).skip_binder().kind()
            {
                return ControlFlow::Break(impl_def_id);
            }
        }

        ControlFlow::Continue(())
    }
}
