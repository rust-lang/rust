use core::ops::ControlFlow;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_hir as hir;
use rustc_infer::infer::{DefineOpaqueTypes, InferOk, TyCtxtInferExt};
use rustc_infer::traits;
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitor, TypingMode, Unnormalized, Upcast,
};
use rustc_span::DUMMY_SP;
use rustc_span::def_id::DefId;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use tracing::{debug, instrument, trace};

use crate::clean;
use crate::clean::{
    clean_middle_assoc_item, clean_middle_ty, clean_trait_ref_with_constraints, clean_ty_generics,
};
use crate::core::DocContext;

/// Detects recursive types to avoid infinite loops in blanket impl evaluation.
fn contains_recursive_type(tcx: TyCtxt<'_>, item_def_id: DefId) -> bool {
    let Some(adt_def) = tcx.type_of(item_def_id).skip_binder().ty_adt_def() else {
        return false;
    };

    struct FindAdtVisitor<'tcx> {
        target: DefId,
        tcx: TyCtxt<'tcx>,
        visited: FxHashSet<DefId>,
    }

    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for FindAdtVisitor<'tcx> {
        type Result = ControlFlow<()>;

        fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
            if let ty::Adt(adt_def, _) = t.kind() {
                if adt_def.did() == self.target {
                    return ControlFlow::Break(());
                }
                if self.visited.insert(adt_def.did()) {
                    for field in adt_def.all_fields() {
                        let field_ty = self.tcx.type_of(field.did).skip_binder();
                        if self.visit_ty(field_ty).is_break() {
                            return ControlFlow::Break(());
                        }
                    }
                }
            }
            t.super_visit_with(self)
        }
    }

    let visited = FxHashSet::default();
    let mut visitor = FindAdtVisitor { target: item_def_id, tcx, visited };
    for field in adt_def.all_fields() {
        let field_ty = tcx.type_of(field.did).skip_binder();
        if visitor.visit_ty(field_ty).is_break() {
            return true;
        }
    }
    false
}

#[instrument(level = "debug", skip(cx))]
pub(crate) fn synthesize_blanket_impls(
    cx: &mut DocContext<'_>,
    item_def_id: DefId,
) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let ty = tcx.type_of(item_def_id);

    if contains_recursive_type(tcx, item_def_id) {
        debug!("skipping blanket impls for recursive type {item_def_id:?}");
        return Vec::new();
    }

    // Keep one infcx for all blanket impls on this type so the trait solver
    // doesn't flush its caches between each one.
    let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
    let args = infcx.fresh_args_for_item(DUMMY_SP, item_def_id);
    let impl_ty = ty.instantiate(tcx, args).skip_norm_wip();

    let mut blanket_impls = Vec::new();
    for trait_def_id in tcx.visible_traits() {
        if !cx.cache.effective_visibilities.is_reachable(tcx, trait_def_id)
            || cx.synthetic_blanket_impls.contains(&(ty.skip_binder(), trait_def_id))
        {
            continue;
        }
        // NOTE: doesn't use `for_each_relevant_impl` to avoid looking at anything besides blanket impls
        let trait_impls = tcx.trait_impls_of(trait_def_id);
        for &impl_def_id in trait_impls.blanket_impls() {
            trace!("considering impl `{impl_def_id:?}` for trait `{trait_def_id:?}`");

            let trait_ref = tcx.impl_trait_ref(impl_def_id);
            if !matches!(trait_ref.skip_binder().self_ty().kind(), ty::Param(_)) {
                continue;
            }

            // Roll back inference state per impl but keep the infcx alive
            // so earlier evaluations still help with later ones.
            let applies = infcx.probe(|_| {
                let impl_args = infcx.fresh_args_for_item(DUMMY_SP, impl_def_id);
                let impl_trait_ref = trait_ref.instantiate(tcx, impl_args).skip_norm_wip();
                let param_env = ty::ParamEnv::empty();

                // Require the type the impl is implemented on to match
                // our type, and ignore the impl if there was a mismatch.
                let Ok(eq_result) = infcx.at(&traits::ObligationCause::dummy(), param_env).eq(
                    DefineOpaqueTypes::Yes,
                    impl_trait_ref.self_ty(),
                    impl_ty,
                ) else {
                    return false;
                };
                let InferOk { value: (), obligations } = eq_result;
                // FIXME(eddyb) ignoring `obligations` might cause false positives.
                drop(obligations);

                let clauses = tcx
                    .clauses_of(impl_def_id)
                    .instantiate(tcx, impl_args)
                    .clauses
                    .into_iter()
                    .map(Unnormalized::skip_norm_wip)
                    .chain(Some(impl_trait_ref.upcast(tcx)));
                for clause in clauses {
                    let obligation = traits::Obligation::new(
                        tcx,
                        traits::ObligationCause::dummy(),
                        param_env,
                        clause,
                    );
                    match infcx.evaluate_obligation(&obligation) {
                        Ok(eval_result) if eval_result.may_apply() => {}
                        Err(traits::OverflowError::Canonical) => {}
                        _ => return false,
                    }
                }
                true
            });

            if !applies {
                continue;
            }
            debug!("found applicable impl for trait ref {trait_ref:?}");

            cx.synthetic_blanket_impls.insert((ty.skip_binder(), trait_def_id));

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
                            ty::Binder::dummy(trait_ref.instantiate_identity().skip_norm_wip()),
                            ThinVec::new(),
                        )),
                        for_: clean_middle_ty(
                            ty::Binder::dummy(ty.instantiate_identity().skip_norm_wip()),
                            cx,
                            None,
                            None,
                        ),
                        items: tcx
                            .associated_items(impl_def_id)
                            .in_definition_order()
                            .filter(|item| !item.is_impl_trait_in_trait())
                            .map(|item| clean_middle_assoc_item(item, cx))
                            .collect(),
                        polarity: ty::ImplPolarity::Positive,
                        kind: clean::ImplKind::Blanket(Box::new(clean_middle_ty(
                            ty::Binder::dummy(
                                trait_ref.instantiate_identity().skip_norm_wip().self_ty(),
                            ),
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
    }

    blanket_impls
}
