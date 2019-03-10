use rustc::hir;
use rustc::traits;
use rustc::ty::ToPredicate;
use rustc::ty::subst::Subst;
use rustc::infer::InferOk;
use syntax_pos::DUMMY_SP;

use crate::core::DocAccessLevels;

use super::*;

use self::def_ctor::{get_def_from_def_id, get_def_from_hir_id};

pub struct BlanketImplFinder<'a, 'tcx> {
    pub cx: &'a core::DocContext<'tcx>,
}

impl<'a, 'tcx> BlanketImplFinder<'a, 'tcx> {
    pub fn new(cx: &'a core::DocContext<'tcx>) -> Self {
        BlanketImplFinder { cx }
    }

    pub fn get_with_def_id(&self, def_id: DefId) -> Vec<Item> {
        get_def_from_def_id(&self.cx, def_id, &|def_ctor| {
            self.get_blanket_impls(def_id, &def_ctor, None)
        })
    }

    pub fn get_with_hir_id(&self, id: hir::HirId, name: String) -> Vec<Item> {
        get_def_from_hir_id(&self.cx, id, name, &|def_ctor, name| {
            let did = self.cx.tcx.hir().local_def_id_from_hir_id(id);
            self.get_blanket_impls(did, &def_ctor, Some(name))
        })
    }

    pub fn get_blanket_impls<F>(
        &self,
        def_id: DefId,
        def_ctor: &F,
        name: Option<String>,
    ) -> Vec<Item>
    where F: Fn(DefId) -> Def {
        debug!("get_blanket_impls(def_id={:?}, ...)", def_id);
        let mut impls = Vec::new();
        if self.cx
            .tcx
            .get_attrs(def_id)
            .lists("doc")
            .has_word("hidden")
        {
            debug!(
                "get_blanket_impls(def_id={:?}, def_ctor=...): item has doc('hidden'), \
                 aborting",
                def_id
            );
            return impls;
        }
        let ty = self.cx.tcx.type_of(def_id);
        let generics = self.cx.tcx.generics_of(def_id);
        let real_name = name.map(|name| Ident::from_str(&name));
        let param_env = self.cx.tcx.param_env(def_id);
        for &trait_def_id in self.cx.all_traits.iter() {
            if !self.cx.renderinfo.borrow().access_levels.is_doc_reachable(trait_def_id) ||
               self.cx.generated_synthetics
                      .borrow_mut()
                      .get(&(def_id, trait_def_id))
                      .is_some() {
                continue
            }
            self.cx.tcx.for_each_relevant_impl(trait_def_id, ty, |impl_def_id| {
                self.cx.tcx.infer_ctxt().enter(|infcx| {
                    debug!("get_blanet_impls: Considering impl for trait '{:?}' {:?}",
                           trait_def_id, impl_def_id);
                    let t_generics = infcx.tcx.generics_of(impl_def_id);
                    let trait_ref = infcx.tcx.impl_trait_ref(impl_def_id)
                                             .expect("Cannot get impl trait");

                    match trait_ref.self_ty().sty {
                        ty::Param(_) => {},
                        _ => return,
                    }

                    let substs = infcx.fresh_substs_for_item(DUMMY_SP, def_id);
                    let ty = ty.subst(infcx.tcx, substs);
                    let param_env = param_env.subst(infcx.tcx, substs);

                    let impl_substs = infcx.fresh_substs_for_item(DUMMY_SP, impl_def_id);
                    let trait_ref = trait_ref.subst(infcx.tcx, impl_substs);

                    // Require the type the impl is implemented on to match
                    // our type, and ignore the impl if there was a mismatch.
                    let cause = traits::ObligationCause::dummy();
                    let eq_result = infcx.at(&cause, param_env)
                                         .eq(trait_ref.self_ty(), ty);
                    if let Ok(InferOk { value: (), obligations }) = eq_result {
                        // FIXME(eddyb) ignoring `obligations` might cause false positives.
                        drop(obligations);

                        debug!(
                            "invoking predicate_may_hold: param_env={:?}, trait_ref={:?}, ty={:?}",
                             param_env, trait_ref, ty
                        );
                        let may_apply = match infcx.evaluate_obligation(
                            &traits::Obligation::new(
                                cause,
                                param_env,
                                trait_ref.to_predicate(),
                            ),
                        ) {
                            Ok(eval_result) => eval_result.may_apply(),
                            Err(traits::OverflowError) => true, // overflow doesn't mean yes *or* no
                        };
                        debug!("get_blanket_impls: found applicable impl: {}\
                               for trait_ref={:?}, ty={:?}",
                               may_apply, trait_ref, ty);

                        if !may_apply {
                            return
                        }
                        self.cx.generated_synthetics.borrow_mut()
                                                    .insert((def_id, trait_def_id));
                        let trait_ = hir::TraitRef {
                            path: get_path_for_type(infcx.tcx,
                                                    trait_def_id,
                                                    hir::def::Def::Trait),
                            hir_ref_id: hir::DUMMY_HIR_ID,
                        };
                        let provided_trait_methods =
                            infcx.tcx.provided_trait_methods(trait_def_id)
                                     .into_iter()
                                     .map(|meth| meth.ident.to_string())
                                     .collect();

                        let ty = self.cx.get_real_ty(def_id, def_ctor, &real_name, generics);
                        let predicates = infcx.tcx.predicates_of(impl_def_id);

                        impls.push(Item {
                            source: infcx.tcx.def_span(impl_def_id).clean(self.cx),
                            name: None,
                            attrs: Default::default(),
                            visibility: None,
                            def_id: self.cx.next_def_id(impl_def_id.krate),
                            stability: None,
                            deprecation: None,
                            inner: ImplItem(Impl {
                                unsafety: hir::Unsafety::Normal,
                                generics: (t_generics, &predicates).clean(self.cx),
                                provided_trait_methods,
                                trait_: Some(trait_.clean(self.cx)),
                                for_: ty.clean(self.cx),
                                items: infcx.tcx.associated_items(impl_def_id)
                                                .collect::<Vec<_>>()
                                                .clean(self.cx),
                                polarity: None,
                                synthetic: false,
                                blanket_impl: Some(infcx.tcx.type_of(impl_def_id)
                                                            .clean(self.cx)),
                            }),
                        });
                    }
                });
            });
        }
        impls
    }
}
