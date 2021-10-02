use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::return_ty;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, HirId};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{Opaque, PredicateKind::Trait};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{sym, Span};
use rustc_trait_selection::traits::error_reporting::suggestions::InferCtxtExt;
use rustc_trait_selection::traits::{self, FulfillmentError, TraitEngine};

declare_clippy_lint! {
    /// ### What it does
    /// This lint requires Future implementations returned from
    /// functions and methods to implement the `Send` marker trait. It is mostly
    /// used by library authors (public and internal) that target an audience where
    /// multithreaded executors are likely to be used for running these Futures.
    ///
    /// ### Why is this bad?
    /// A Future implementation captures some state that it
    /// needs to eventually produce its final value. When targeting a multithreaded
    /// executor (which is the norm on non-embedded devices) this means that this
    /// state may need to be transported to other threads, in other words the
    /// whole Future needs to implement the `Send` marker trait. If it does not,
    /// then the resulting Future cannot be submitted to a thread pool in the
    /// end user’s code.
    ///
    /// Especially for generic functions it can be confusing to leave the
    /// discovery of this problem to the end user: the reported error location
    /// will be far from its cause and can in many cases not even be fixed without
    /// modifying the library where the offending Future implementation is
    /// produced.
    ///
    /// ### Example
    /// ```rust
    /// async fn not_send(bytes: std::rc::Rc<[u8]>) {}
    /// ```
    /// Use instead:
    /// ```rust
    /// async fn is_send(bytes: std::sync::Arc<[u8]>) {}
    /// ```
    pub FUTURE_NOT_SEND,
    nursery,
    "public Futures must be Send"
}

declare_lint_pass!(FutureNotSend => [FUTURE_NOT_SEND]);

impl<'tcx> LateLintPass<'tcx> for FutureNotSend {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'tcx>,
        _: &'tcx Body<'tcx>,
        _: Span,
        hir_id: HirId,
    ) {
        if let FnKind::Closure = kind {
            return;
        }
        let ret_ty = return_ty(cx, hir_id);
        if let Opaque(id, subst) = *ret_ty.kind() {
            let preds = cx.tcx.explicit_item_bounds(id);
            let mut is_future = false;
            for &(p, _span) in preds {
                let p = p.subst(cx.tcx, subst);
                if let Some(trait_ref) = p.to_opt_poly_trait_ref() {
                    if Some(trait_ref.value.def_id()) == cx.tcx.lang_items().future_trait() {
                        is_future = true;
                        break;
                    }
                }
            }
            if is_future {
                let send_trait = cx.tcx.get_diagnostic_item(sym::Send).unwrap();
                let span = decl.output.span();
                let send_result = cx.tcx.infer_ctxt().enter(|infcx| {
                    let cause = traits::ObligationCause::misc(span, hir_id);
                    let mut fulfillment_cx = traits::FulfillmentContext::new();
                    fulfillment_cx.register_bound(&infcx, cx.param_env, ret_ty, send_trait, cause);
                    fulfillment_cx.select_all_or_error(&infcx)
                });
                if let Err(send_errors) = send_result {
                    span_lint_and_then(
                        cx,
                        FUTURE_NOT_SEND,
                        span,
                        "future cannot be sent between threads safely",
                        |db| {
                            cx.tcx.infer_ctxt().enter(|infcx| {
                                for FulfillmentError { obligation, .. } in send_errors {
                                    infcx.maybe_note_obligation_cause_for_async_await(db, &obligation);
                                    if let Trait(trait_pred) = obligation.predicate.kind().skip_binder() {
                                        db.note(&format!(
                                            "`{}` doesn't implement `{}`",
                                            trait_pred.self_ty(),
                                            trait_pred.trait_ref.print_only_trait_path(),
                                        ));
                                    }
                                }
                            });
                        },
                    );
                }
            }
        }
    }
}
