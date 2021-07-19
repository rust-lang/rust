use crate::collect::ItemCtxt;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::HirId;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::ObligationCause;
use rustc_infer::traits::TraitEngine;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, ToPredicate, TyCtxt};
use rustc_trait_selection::traits;

pub fn provide(providers: &mut Providers) {
    *providers = Providers { diagnostic_hir_wf_check, ..*providers };
}

// Ideally, this would be in `rustc_trait_selection`, but we
// need access to `ItemCtxt`
fn diagnostic_hir_wf_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    (predicate, hir_id): (ty::Predicate<'tcx>, HirId),
) -> Option<ObligationCause<'tcx>> {
    let hir = tcx.hir();
    // HIR wfcheck should only ever happen as part of improving an existing error
    tcx.sess.delay_span_bug(hir.span(hir_id), "Performed HIR wfcheck without an existing error!");

    // Currently, we only handle WF checking for items (e.g. associated items).
    // It would be nice to extend this to handle wf checks inside functions.
    let def_id = match tcx.hir().opt_local_def_id(hir_id) {
        Some(def_id) => def_id,
        None => return None,
    };

    // FIXME - figure out how we want to handle wf-checking for
    // things inside a function body.
    let icx = ItemCtxt::new(tcx, def_id.to_def_id());

    // To perform HIR-based WF checking, we iterate over all HIR types
    // that occur 'inside' the item we're checking. For example,
    // given the type `Option<MyStruct<u8>>`, we will check
    // `Option<MyStruct<u8>>`, `MyStruct<u8>`, and `u8`.
    // For each type, we perform a well-formed check, and see if we get
    // an erorr that matches our expected predicate. We keep save
    // the `ObligationCause` corresponding to the *innermost* type,
    // which is the most specific type that we can point to.
    // In general, the different components of an `hir::Ty` may have
    // completely differentr spans due to macro invocations. Pointing
    // to the most accurate part of the type can be the difference
    // between a useless span (e.g. the macro invocation site)
    // and a useful span (e.g. a user-provided type passed in to the macro).
    //
    // This approach is quite inefficient - we redo a lot of work done
    // by the normal WF checker. However, this code is run at most once
    // per reported error - it will have no impact when compilation succeeds,
    // and should only have an impact if a very large number of errors are
    // displaydd to the user.
    struct HirWfCheck<'tcx> {
        tcx: TyCtxt<'tcx>,
        predicate: ty::Predicate<'tcx>,
        cause: Option<ObligationCause<'tcx>>,
        cause_depth: usize,
        icx: ItemCtxt<'tcx>,
        hir_id: HirId,
        param_env: ty::ParamEnv<'tcx>,
        depth: usize,
    }

    impl<'tcx> Visitor<'tcx> for HirWfCheck<'tcx> {
        type Map = intravisit::ErasedMap<'tcx>;
        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }
        fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
            self.tcx.infer_ctxt().enter(|infcx| {
                let mut fulfill = traits::FulfillmentContext::new();
                let tcx_ty = self.icx.to_ty(ty);
                let cause = traits::ObligationCause::new(
                    ty.span,
                    self.hir_id,
                    traits::ObligationCauseCode::MiscObligation,
                );
                fulfill.register_predicate_obligation(
                    &infcx,
                    traits::Obligation::new(
                        cause,
                        self.param_env,
                        ty::PredicateKind::WellFormed(tcx_ty.into()).to_predicate(self.tcx),
                    ),
                );

                if let Err(errors) = fulfill.select_all_or_error(&infcx) {
                    tracing::debug!("Wf-check got errors for {:?}: {:?}", ty, errors);
                    for error in errors {
                        if error.obligation.predicate == self.predicate {
                            // Save the cause from the greatest depth - this corresponds
                            // to picking more-specific types (e.g. `MyStruct<u8>`)
                            // over less-specific types (e.g. `Option<MyStruct<u8>>`)
                            if self.depth >= self.cause_depth {
                                self.cause = Some(error.obligation.cause);
                                self.cause_depth = self.depth
                            }
                        }
                    }
                }
            });
            self.depth += 1;
            intravisit::walk_ty(self, ty);
            self.depth -= 1;
        }
    }

    let mut visitor = HirWfCheck {
        tcx,
        predicate,
        cause: None,
        cause_depth: 0,
        icx,
        hir_id,
        param_env: tcx.param_env(def_id.to_def_id()),
        depth: 0,
    };

    let ty = match tcx.hir().get(hir_id) {
        hir::Node::ImplItem(item) => match item.kind {
            hir::ImplItemKind::TyAlias(ref ty) => Some(ty),
            _ => None,
        },
        _ => None,
    };
    if let Some(ty) = ty {
        visitor.visit_ty(ty);
    }
    visitor.cause
}
