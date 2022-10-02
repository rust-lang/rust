use hir::def_id::LocalDefId;
use rustc_hir as hir;
use rustc_infer::infer::{InferCtxt, TyCtxtInferExt};
use rustc_macros::LintDiagnostic;
use rustc_middle::ty::{
    self, fold::BottomUpFolder, Ty, TypeFoldable, TypeSuperVisitable, TypeVisitable, TypeVisitor,
};
use rustc_span::Span;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;

use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `rpit_hidden_inferred_bound` lint detects cases in which nested RPITs
    /// in associated type bounds are not written generally enough to satisfy the
    /// bounds of the associated type. This functionality was removed in #97346,
    /// but then rolled back in #99860 because it was made into a hard error too
    /// quickly.
    ///
    /// We plan on reintroducing this as a hard error, but in the mean time, this
    /// lint serves to warn and suggest fixes for any use-cases which rely on this
    /// behavior.
    pub RPIT_HIDDEN_INFERRED_BOUND,
    Warn,
    "detects the use of nested RPITs in associated type bounds that are not general enough"
}

declare_lint_pass!(RpitHiddenInferredBound => [RPIT_HIDDEN_INFERRED_BOUND]);

impl<'tcx> LateLintPass<'tcx> for RpitHiddenInferredBound {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: hir::intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        _: &'tcx hir::Body<'tcx>,
        _: rustc_span::Span,
        id: hir::HirId,
    ) {
        if matches!(kind, hir::intravisit::FnKind::Closure) {
            return;
        }

        let fn_def_id = cx.tcx.hir().local_def_id(id);
        let sig: ty::FnSig<'tcx> =
            cx.tcx.liberate_late_bound_regions(fn_def_id.to_def_id(), cx.tcx.fn_sig(fn_def_id));
        cx.tcx.infer_ctxt().enter(|ref infcx| {
            sig.output().visit_with(&mut VisitOpaqueBounds { infcx, cx, fn_def_id });
        });
    }
}

struct VisitOpaqueBounds<'a, 'cx, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    cx: &'cx LateContext<'tcx>,
    fn_def_id: LocalDefId,
}

impl<'tcx> TypeVisitor<'tcx> for VisitOpaqueBounds<'_, '_, 'tcx> {
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> std::ops::ControlFlow<Self::BreakTy> {
        if let ty::Opaque(def_id, substs) = *ty.kind()
            && let Some(hir::Node::Item(item)) = self.cx.tcx.hir().get_if_local(def_id)
            && let hir::ItemKind::OpaqueTy(opaque) = &item.kind
            && let hir::OpaqueTyOrigin::FnReturn(origin_def_id) = opaque.origin
            && origin_def_id == self.fn_def_id
        {
            for pred_and_span in self.cx.tcx.bound_explicit_item_bounds(def_id).transpose_iter() {
                let pred_span = pred_and_span.0.1;
                let predicate = self.cx.tcx.liberate_late_bound_regions(
                    def_id,
                    pred_and_span.map_bound(|(pred, _)| *pred).subst(self.cx.tcx, substs).kind(),
                );
                let ty::PredicateKind::Projection(proj) = predicate else {
                    continue;
                };
                let Some(proj_term) = proj.term.ty() else { continue };

                let proj_ty = self
                    .cx
                    .tcx
                    .mk_projection(proj.projection_ty.item_def_id, proj.projection_ty.substs);
                let proj_replacer = &mut BottomUpFolder {
                    tcx: self.cx.tcx,
                    ty_op: |ty| if ty == proj_ty { proj_term } else { ty },
                    lt_op: |lt| lt,
                    ct_op: |ct| ct,
                };
                for assoc_pred_and_span in self
                    .cx
                    .tcx
                    .bound_explicit_item_bounds(proj.projection_ty.item_def_id)
                    .transpose_iter()
                {
                    let assoc_pred_span = assoc_pred_and_span.0.1;
                    let assoc_pred = assoc_pred_and_span
                        .map_bound(|(pred, _)| *pred)
                        .subst(self.cx.tcx, &proj.projection_ty.substs)
                        .fold_with(proj_replacer);
                    if !self.infcx.predicate_must_hold_modulo_regions(&traits::Obligation::new(
                        traits::ObligationCause::dummy(),
                        self.cx.param_env,
                        assoc_pred,
                    )) {
                        let (suggestion, suggest_span) =
                            match (proj_term.kind(), assoc_pred.kind().skip_binder()) {
                                (ty::Opaque(def_id, _), ty::PredicateKind::Trait(trait_pred)) => (
                                    format!(" + {}", trait_pred.print_modifiers_and_trait_path()),
                                    Some(self.cx.tcx.def_span(def_id).shrink_to_hi()),
                                ),
                                _ => (String::new(), None),
                            };
                        self.cx.emit_spanned_lint(
                            RPIT_HIDDEN_INFERRED_BOUND,
                            pred_span,
                            RpitHiddenInferredBoundLint {
                                ty,
                                proj_ty: proj_term,
                                assoc_pred_span,
                                suggestion,
                                suggest_span,
                            },
                        );
                    }
                }
            }
        }

        ty.super_visit_with(self)
    }
}

#[derive(LintDiagnostic)]
#[diag(lint::rpit_hidden_inferred_bound)]
struct RpitHiddenInferredBoundLint<'tcx> {
    ty: Ty<'tcx>,
    proj_ty: Ty<'tcx>,
    #[label(lint::specifically)]
    assoc_pred_span: Span,
    #[suggestion_verbose(applicability = "machine-applicable", code = "{suggestion}")]
    suggest_span: Option<Span>,
    suggestion: String,
}
