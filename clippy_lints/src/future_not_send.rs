use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::return_ty;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, AliasTy, ClauseKind, PredicateKind};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::{sym, Span};
use rustc_trait_selection::traits::error_reporting::suggestions::TypeErrCtxtExt;
use rustc_trait_selection::traits::{self, FulfillmentError, ObligationCtxt};

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
    #[clippy::version = "1.44.0"]
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
        fn_def_id: LocalDefId,
    ) {
        if let FnKind::Closure = kind {
            return;
        }
        let ret_ty = return_ty(cx, cx.tcx.hir().local_def_id_to_hir_id(fn_def_id).expect_owner());
        if let ty::Alias(ty::Opaque, AliasTy { def_id, args, .. }) = *ret_ty.kind() {
            let preds = cx.tcx.explicit_item_bounds(def_id);
            let mut is_future = false;
            for (p, _span) in preds.iter_instantiated_copied(cx.tcx, args) {
                if let Some(trait_pred) = p.as_trait_clause() {
                    if Some(trait_pred.skip_binder().trait_ref.def_id) == cx.tcx.lang_items().future_trait() {
                        is_future = true;
                        break;
                    }
                }
            }
            if is_future {
                let send_trait = cx.tcx.get_diagnostic_item(sym::Send).unwrap();
                let span = decl.output.span();
                let infcx = cx.tcx.infer_ctxt().build();
                let ocx = ObligationCtxt::new(&infcx);
                let cause = traits::ObligationCause::misc(span, fn_def_id);
                ocx.register_bound(cause, cx.param_env, ret_ty, send_trait);
                let send_errors = ocx.select_all_or_error();
                if !send_errors.is_empty() {
                    span_lint_and_then(
                        cx,
                        FUTURE_NOT_SEND,
                        span,
                        "future cannot be sent between threads safely",
                        |db| {
                            for FulfillmentError { obligation, .. } in send_errors {
                                infcx
                                    .err_ctxt()
                                    .maybe_note_obligation_cause_for_async_await(db, &obligation);
                                if let PredicateKind::Clause(ClauseKind::Trait(trait_pred)) =
                                    obligation.predicate.kind().skip_binder()
                                {
                                    db.note(format!(
                                        "`{}` doesn't implement `{}`",
                                        trait_pred.self_ty(),
                                        trait_pred.trait_ref.print_only_trait_path(),
                                    ));
                                }
                            }
                        },
                    );
                }
            }
        }
    }
}
