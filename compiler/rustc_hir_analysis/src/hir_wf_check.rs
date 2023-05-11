use crate::collect::ItemCtxt;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{ForeignItem, ForeignItemKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{ObligationCause, WellFormedLoc};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, Region, TyCtxt, TypeFoldable, TypeFolder};
use rustc_span::def_id::LocalDefId;
use rustc_trait_selection::traits::{self, ObligationCtxt};

pub fn provide(providers: &mut Providers) {
    *providers = Providers { diagnostic_hir_wf_check, ..*providers };
}

// Ideally, this would be in `rustc_trait_selection`, but we
// need access to `ItemCtxt`
fn diagnostic_hir_wf_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    (predicate, loc): (ty::Predicate<'tcx>, WellFormedLoc),
) -> Option<ObligationCause<'tcx>> {
    let hir = tcx.hir();

    let def_id = match loc {
        WellFormedLoc::Ty(def_id) => def_id,
        WellFormedLoc::Param { function, param_idx: _ } => function,
    };
    let hir_id = hir.local_def_id_to_hir_id(def_id);

    // HIR wfcheck should only ever happen as part of improving an existing error
    tcx.sess
        .delay_span_bug(tcx.def_span(def_id), "Performed HIR wfcheck without an existing error!");

    let icx = ItemCtxt::new(tcx, def_id);

    // To perform HIR-based WF checking, we iterate over all HIR types
    // that occur 'inside' the item we're checking. For example,
    // given the type `Option<MyStruct<u8>>`, we will check
    // `Option<MyStruct<u8>>`, `MyStruct<u8>`, and `u8`.
    // For each type, we perform a well-formed check, and see if we get
    // an error that matches our expected predicate. We save
    // the `ObligationCause` corresponding to the *innermost* type,
    // which is the most specific type that we can point to.
    // In general, the different components of an `hir::Ty` may have
    // completely different spans due to macro invocations. Pointing
    // to the most accurate part of the type can be the difference
    // between a useless span (e.g. the macro invocation site)
    // and a useful span (e.g. a user-provided type passed into the macro).
    //
    // This approach is quite inefficient - we redo a lot of work done
    // by the normal WF checker. However, this code is run at most once
    // per reported error - it will have no impact when compilation succeeds,
    // and should only have an impact if a very large number of errors is
    // displayed to the user.
    struct HirWfCheck<'tcx> {
        tcx: TyCtxt<'tcx>,
        predicate: ty::Predicate<'tcx>,
        cause: Option<ObligationCause<'tcx>>,
        cause_depth: usize,
        icx: ItemCtxt<'tcx>,
        def_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        depth: usize,
    }

    impl<'tcx> Visitor<'tcx> for HirWfCheck<'tcx> {
        fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx>) {
            let infcx = self.tcx.infer_ctxt().build();
            let ocx = ObligationCtxt::new(&infcx);

            let tcx_ty = self.icx.to_ty(ty).fold_with(&mut EraseAllBoundRegions { tcx: self.tcx });
            let cause = traits::ObligationCause::new(
                ty.span,
                self.def_id,
                traits::ObligationCauseCode::WellFormed(None),
            );

            ocx.register_obligation(traits::Obligation::new(
                self.tcx,
                cause,
                self.param_env,
                ty::PredicateKind::WellFormed(tcx_ty.into()),
            ));

            for error in ocx.select_all_or_error() {
                debug!("Wf-check got error for {:?}: {:?}", ty, error);
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
        def_id,
        param_env: tcx.param_env(def_id.to_def_id()),
        depth: 0,
    };

    // Get the starting `hir::Ty` using our `WellFormedLoc`.
    // We will walk 'into' this type to try to find
    // a more precise span for our predicate.
    let tys = match loc {
        WellFormedLoc::Ty(_) => match hir.get(hir_id) {
            hir::Node::ImplItem(item) => match item.kind {
                hir::ImplItemKind::Type(ty) => vec![ty],
                hir::ImplItemKind::Const(ty, _) => vec![ty],
                ref item => bug!("Unexpected ImplItem {:?}", item),
            },
            hir::Node::TraitItem(item) => match item.kind {
                hir::TraitItemKind::Type(_, ty) => ty.into_iter().collect(),
                hir::TraitItemKind::Const(ty, _) => vec![ty],
                ref item => bug!("Unexpected TraitItem {:?}", item),
            },
            hir::Node::Item(item) => match item.kind {
                hir::ItemKind::Static(ty, _, _) | hir::ItemKind::Const(ty, _) => vec![ty],
                hir::ItemKind::Impl(impl_) => match &impl_.of_trait {
                    Some(t) => t
                        .path
                        .segments
                        .last()
                        .iter()
                        .flat_map(|seg| seg.args().args)
                        .filter_map(|arg| {
                            if let hir::GenericArg::Type(ty) = arg { Some(*ty) } else { None }
                        })
                        .chain([impl_.self_ty])
                        .collect(),
                    None => {
                        vec![impl_.self_ty]
                    }
                },
                ref item => bug!("Unexpected item {:?}", item),
            },
            hir::Node::Field(field) => vec![field.ty],
            hir::Node::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Static(ty, _), ..
            }) => vec![*ty],
            hir::Node::GenericParam(hir::GenericParam {
                kind: hir::GenericParamKind::Type { default: Some(ty), .. },
                ..
            }) => vec![*ty],
            ref node => bug!("Unexpected node {:?}", node),
        },
        WellFormedLoc::Param { function: _, param_idx } => {
            let fn_decl = hir.fn_decl_by_hir_id(hir_id).unwrap();
            // Get return type
            if param_idx as usize == fn_decl.inputs.len() {
                match fn_decl.output {
                    hir::FnRetTy::Return(ty) => vec![ty],
                    // The unit type `()` is always well-formed
                    hir::FnRetTy::DefaultReturn(_span) => vec![],
                }
            } else {
                vec![&fn_decl.inputs[param_idx as usize]]
            }
        }
    };
    for ty in tys {
        visitor.visit_ty(ty);
    }
    visitor.cause
}

struct EraseAllBoundRegions<'tcx> {
    tcx: TyCtxt<'tcx>,
}

// Higher ranked regions are complicated.
// To make matters worse, the HIR WF check can instantiate them
// outside of a `Binder`, due to the way we (ab)use
// `ItemCtxt::to_ty`. To make things simpler, we just erase all
// of them, regardless of depth. At worse, this will give
// us an inaccurate span for an error message, but cannot
// lead to unsoundness (we call `delay_span_bug` at the start
// of `diagnostic_hir_wf_check`).
impl<'tcx> TypeFolder<TyCtxt<'tcx>> for EraseAllBoundRegions<'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        if r.is_late_bound() { self.tcx.lifetimes.re_erased } else { r }
    }
}
