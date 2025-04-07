use rustc_hir::intravisit::{self, Visitor, VisitorExt};
use rustc_hir::{self as hir, AmbigArg, ForeignItem, ForeignItemKind};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{ObligationCause, WellFormedLoc};
use rustc_middle::bug;
use rustc_middle::query::Providers;
use rustc_middle::ty::{self, TyCtxt, TypingMode, fold_regions};
use rustc_span::def_id::LocalDefId;
use rustc_trait_selection::traits::{self, ObligationCtxt};
use tracing::debug;

use crate::collect::ItemCtxt;

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { diagnostic_hir_wf_check, ..*providers };
}

// Ideally, this would be in `rustc_trait_selection`, but we
// need access to `ItemCtxt`
fn diagnostic_hir_wf_check<'tcx>(
    tcx: TyCtxt<'tcx>,
    (predicate, loc): (ty::Predicate<'tcx>, WellFormedLoc),
) -> Option<ObligationCause<'tcx>> {
    let def_id = match loc {
        WellFormedLoc::Ty(def_id) => def_id,
        WellFormedLoc::Param { function, param_idx: _ } => function,
    };
    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    // HIR wfcheck should only ever happen as part of improving an existing error
    tcx.dcx()
        .span_delayed_bug(tcx.def_span(def_id), "Performed HIR wfcheck without an existing error!");

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
        fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx, AmbigArg>) {
            let infcx = self.tcx.infer_ctxt().build(TypingMode::non_body_analysis());
            let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

            // We don't handle infer vars but we wouldn't handle them anyway as we're creating a
            // fresh `InferCtxt` in this function.
            let tcx_ty = self.icx.lower_ty(ty.as_unambig_ty());
            // This visitor can walk into binders, resulting in the `tcx_ty` to
            // potentially reference escaping bound variables. We simply erase
            // those here.
            let tcx_ty = fold_regions(self.tcx, tcx_ty, |r, _| {
                if r.is_bound() { self.tcx.lifetimes.re_erased } else { r }
            });
            let cause = traits::ObligationCause::new(
                ty.span,
                self.def_id,
                traits::ObligationCauseCode::WellFormed(None),
            );

            ocx.register_obligation(traits::Obligation::new(
                self.tcx,
                cause,
                self.param_env,
                ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(tcx_ty.into())),
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
        WellFormedLoc::Ty(_) => match tcx.hir_node(hir_id) {
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
                hir::ItemKind::TyAlias(_, ty, _)
                | hir::ItemKind::Static(_, ty, _, _)
                | hir::ItemKind::Const(_, ty, _, _) => vec![ty],
                hir::ItemKind::Impl(impl_) => match &impl_.of_trait {
                    Some(t) => t
                        .path
                        .segments
                        .last()
                        .iter()
                        .flat_map(|seg| seg.args().args)
                        .filter_map(|arg| {
                            if let hir::GenericArg::Type(ty) = arg {
                                Some(ty.as_unambig_ty())
                            } else {
                                None
                            }
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
                kind: ForeignItemKind::Static(ty, _, _), ..
            }) => vec![*ty],
            hir::Node::GenericParam(hir::GenericParam {
                kind: hir::GenericParamKind::Type { default: Some(ty), .. },
                ..
            }) => vec![*ty],
            hir::Node::AnonConst(_) => {
                if let Some(const_param_id) = tcx.hir_opt_const_param_default_param_def_id(hir_id)
                    && let hir::Node::GenericParam(hir::GenericParam {
                        kind: hir::GenericParamKind::Const { ty, .. },
                        ..
                    }) = tcx.hir_node_by_def_id(const_param_id)
                {
                    vec![*ty]
                } else {
                    vec![]
                }
            }
            ref node => bug!("Unexpected node {:?}", node),
        },
        WellFormedLoc::Param { function: _, param_idx } => {
            let fn_decl = tcx.hir_fn_decl_by_hir_id(hir_id).unwrap();
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
        visitor.visit_ty_unambig(ty);
    }
    visitor.cause
}
