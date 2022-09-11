use crate::rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_hir as hir;
use rustc_infer::infer::{InferOk, TyCtxtInferExt};
use rustc_infer::traits;
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::ToPredicate;
use rustc_span::DUMMY_SP;

use super::*;

pub(crate) struct BlanketImplFinder<'a, 'tcx> {
    pub(crate) cx: &'a mut core::DocContext<'tcx>,
}

impl<'a, 'tcx> BlanketImplFinder<'a, 'tcx> {
    pub(crate) fn get_blanket_impls(&mut self, item_def_id: DefId) -> Vec<Item> {
        let param_env = self.cx.tcx.param_env(item_def_id);
        let ty = self.cx.tcx.bound_type_of(item_def_id);

        trace!("get_blanket_impls({:?})", ty);
        let mut impls = Vec::new();
        self.cx.with_all_traits(|cx, all_traits| {
            for &trait_def_id in all_traits {
                if !cx.cache.access_levels.is_public(trait_def_id)
                    || cx.generated_synthetics.get(&(ty.0, trait_def_id)).is_some()
                {
                    continue;
                }
                // NOTE: doesn't use `for_each_relevant_impl` to avoid looking at anything besides blanket impls
                let trait_impls = cx.tcx.trait_impls_of(trait_def_id);
                for &impl_def_id in trait_impls.blanket_impls() {
                    trace!(
                        "get_blanket_impls: Considering impl for trait '{:?}' {:?}",
                        trait_def_id,
                        impl_def_id
                    );
                    let trait_ref = cx.tcx.bound_impl_trait_ref(impl_def_id).unwrap();
                    let is_param = matches!(trait_ref.0.self_ty().kind(), ty::Param(_));
                    let may_apply = is_param && cx.tcx.infer_ctxt().enter(|infcx| {
                        let substs = infcx.fresh_substs_for_item(DUMMY_SP, item_def_id);
                        let ty = ty.subst(infcx.tcx, substs);
                        let param_env = EarlyBinder(param_env).subst(infcx.tcx, substs);

                        let impl_substs = infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
                        let trait_ref = trait_ref.subst(infcx.tcx, impl_substs);

                        // Require the type the impl is implemented on to match
                        // our type, and ignore the impl if there was a mismatch.
                        let cause = traits::ObligationCause::dummy();
                        let eq_result = infcx.at(&cause, param_env).eq(trait_ref.self_ty(), ty);
                        if let Ok(InferOk { value: (), obligations }) = eq_result {
                            // FIXME(eddyb) ignoring `obligations` might cause false positives.
                            drop(obligations);

                            trace!(
                                "invoking predicate_may_hold: param_env={:?}, trait_ref={:?}, ty={:?}",
                                param_env,
                                trait_ref,
                                ty
                            );
                            let predicates = cx
                                .tcx
                                .predicates_of(impl_def_id)
                                .instantiate(cx.tcx, impl_substs)
                                .predicates
                                .into_iter()
                                .chain(Some(
                                    ty::Binder::dummy(trait_ref)
                                        .to_poly_trait_predicate()
                                        .map_bound(ty::PredicateKind::Trait)
                                        .to_predicate(infcx.tcx),
                                ));
                            for predicate in predicates {
                                debug!("testing predicate {:?}", predicate);
                                let obligation = traits::Obligation::new(
                                    traits::ObligationCause::dummy(),
                                    param_env,
                                    predicate,
                                );
                                match infcx.evaluate_obligation(&obligation) {
                                    Ok(eval_result) if eval_result.may_apply() => {}
                                    Err(traits::OverflowError::Canonical) => {}
                                    Err(traits::OverflowError::ErrorReporting) => {}
                                    _ => {
                                        return false;
                                    }
                                }
                            }
                            true
                        } else {
                            false
                        }
                    });
                    debug!(
                        "get_blanket_impls: found applicable impl: {} for trait_ref={:?}, ty={:?}",
                        may_apply, trait_ref, ty
                    );
                    if !may_apply {
                        continue;
                    }

                    cx.generated_synthetics.insert((ty.0, trait_def_id));

                    impls.push(Item {
                        name: None,
                        attrs: Default::default(),
                        visibility: Inherited,
                        item_id: ItemId::Blanket { impl_id: impl_def_id, for_: item_def_id },
                        kind: Box::new(ImplItem(Box::new(Impl {
                            unsafety: hir::Unsafety::Normal,
                            generics: clean_ty_generics(
                                cx,
                                cx.tcx.generics_of(impl_def_id),
                                cx.tcx.explicit_predicates_of(impl_def_id),
                            ),
                            // FIXME(eddyb) compute both `trait_` and `for_` from
                            // the post-inference `trait_ref`, as it's more accurate.
                            trait_: Some(clean_trait_ref_with_bindings(cx, trait_ref.0, ThinVec::new())),
                            for_: clean_middle_ty(ty.0, cx, None),
                            items: cx.tcx
                                .associated_items(impl_def_id)
                                .in_definition_order()
                                .map(|x| clean_middle_assoc_item(x, cx))
                                .collect::<Vec<_>>(),
                            polarity: ty::ImplPolarity::Positive,
                            kind: ImplKind::Blanket(Box::new(clean_middle_ty(trait_ref.0.self_ty(), cx, None))),
                        }))),
                        cfg: None,
                    });
                }
            }
        });

        impls
    }
}
