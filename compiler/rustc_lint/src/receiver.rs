use rustc_hir::def_id::DefId;
use rustc_hir::{ImplItemImplKind, ImplItemKind, LangItem, PatKind};
use rustc_infer::infer::InferCtxt;
use rustc_middle::ty::{self, AliasTy, AliasTyKind, ParamEnv, Ty, TyCtxt, TypingEnv, TypingMode};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{DUMMY_SP, kw, sym};
use rustc_trait_selection::infer::{InferCtxtExt, TyCtxtInferExt};

use crate::{LateLintPass, LintContext};

declare_lint! {
    INHERENT_METHOD_ON_RECEIVER,
    Warn,
    "inherent methods on types that implement `Deref` or `Receiver` shadow methods of their target",
}

declare_lint_pass!(InherentMethodOnReceiver => [INHERENT_METHOD_ON_RECEIVER]);

// FIXME:
// - make this lint nightly-only

impl<'tcx> LateLintPass<'tcx> for InherentMethodOnReceiver {
    fn check_impl_item(
        &mut self,
        cx: &crate::LateContext<'tcx>,
        item: &'tcx rustc_hir::ImplItem<'tcx>,
    ) {
        let ImplItemImplKind::Inherent { .. } = item.impl_kind else {
            return;
        };
        if !cx.tcx.effective_visibilities(()).is_exported(item.owner_id.def_id) {
            return;
        }
        let Some(impl_id) = cx.tcx.inherent_impl_of_assoc(item.owner_id.def_id.to_def_id()) else {
            return;
        };
        let ImplItemKind::Fn(signature, body) = item.kind else {
            return;
        };

        let Some(first_param) = cx.tcx.hir_body(body).params.first() else {
            return;
        };

        if !signature.decl.implicit_self.has_implicit_self() {
            let PatKind::Binding(_, _, ident, None) = first_param.pat.kind else {
                return;
            };
            if ident.name != kw::SelfLower {
                return;
            }
        }
        let self_ty: ty::EarlyBinder<'tcx, Ty<'tcx>> = cx.tcx.type_of(impl_id);
        let self_ty = self_ty.instantiate_identity();
        let infcx: InferCtxt<'tcx> = cx.tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        if let Some(CheckResult { impl_plus_target }) =
            check(cx.tcx, &infcx, cx.typing_env(), cx.param_env, self_ty)
        {
            cx.span_lint(INHERENT_METHOD_ON_RECEIVER, first_param.span, |lint| {
                for (impl_id, target_id) in impl_plus_target {
                    lint.span_label(cx.tcx.def_span(impl_id), "trait implemented here");
                    lint.span_label(cx.tcx.def_span(target_id), "with `Target` set here");
                }
                lint.primary_message(
                    "inherent methods on types that implement `Deref` or `Receiver` shadow \
                    methods of their target",
                );
            });
        }
    }
}

struct CheckResult {
    impl_plus_target: Vec<(DefId, DefId)>,
}

fn check<'tcx>(
    tcx: TyCtxt<'tcx>,
    infcx: &InferCtxt<'tcx>,
    typing_env: TypingEnv<'tcx>,
    param_env: ParamEnv<'tcx>,
    ty: Ty<'tcx>,
) -> Option<CheckResult> {
    let deref_trait = tcx.require_lang_item(LangItem::Deref, DUMMY_SP);
    let deref_target = tcx.require_lang_item(LangItem::DerefTarget, DUMMY_SP);
    let receiver_trait = tcx.require_lang_item(LangItem::Receiver, DUMMY_SP);
    let receiver_target = tcx.require_lang_item(LangItem::ReceiverTarget, DUMMY_SP);

    if infcx.type_implements_trait(deref_trait, [ty], param_env).must_apply_modulo_regions() {
        let target_ty = tcx.normalize_erasing_regions(
            typing_env,
            Ty::new_alias(tcx, AliasTyKind::Projection, AliasTy::new(tcx, deref_target, [ty])),
        );
        if let ty::Param(_) = target_ty.kind() {
            let (impl_id, target_id) = find_impl_and_target_id(tcx, deref_trait, ty);
            return Some(CheckResult { impl_plus_target: vec![(impl_id, target_id)] });
        }
        if let Some(mut result) = check(tcx, infcx, typing_env, param_env, target_ty) {
            let (impl_id, target_id) = find_impl_and_target_id(tcx, deref_trait, ty);
            result.impl_plus_target.push((impl_id, target_id));
            return Some(result);
        }
    }
    if tcx.features().arbitrary_self_types()
        && infcx.type_implements_trait(receiver_trait, [ty], param_env).must_apply_modulo_regions()
    {
        let target_ty = tcx.normalize_erasing_regions(
            typing_env,
            Ty::new_alias(tcx, AliasTyKind::Projection, AliasTy::new(tcx, receiver_target, [ty])),
        );
        if let ty::Param(_) = target_ty.kind() {
            let (impl_id, target_id) = find_impl_and_target_id(tcx, receiver_trait, ty);
            return Some(CheckResult { impl_plus_target: vec![(impl_id, target_id)] });
        }
        if let Some(mut result) = check(tcx, infcx, typing_env, param_env, target_ty) {
            let (impl_id, target_id) = find_impl_and_target_id(tcx, receiver_trait, ty);
            result.impl_plus_target.push((impl_id, target_id));
            return Some(result);
        }
    }
    None
}

fn find_impl_and_target_id<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_id: DefId,
    ty: Ty<'tcx>,
) -> (DefId, DefId) {
    let mut impl_id: Option<DefId> = None;
    tcx.for_each_relevant_impl(trait_id, ty, |did| {
        if let Some(_impl_id) = impl_id.take() {
            // let spans = vec![tcx.def_span(impl_id)];
            // println!("{impl_id:?} {did:?}");
            // span_bug!(spans, "found two impl blocks for the trait");
        }
        impl_id = Some(did);
    });
    let impl_id = impl_id.expect("trait not implemented?");

    let targets: Vec<_> =
        tcx.associated_items(impl_id).filter_by_name_unhygienic(sym::Target).collect();
    //assert!(targets.len() == 1, "did not find exactly one target");
    let target_id = targets[0].def_id;
    (impl_id, target_id)
}
