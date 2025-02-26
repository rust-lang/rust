use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::return_ty;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::print::PrintTraitRefExt;
use rustc_middle::ty::{
    self, AliasTy, Binder, ClauseKind, PredicateKind, Ty, TyCtxt, TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::traits::{self, FulfillmentError, ObligationCtxt};

declare_clippy_lint! {
    /// ### What it does
    /// This lint requires Future implementations returned from
    /// functions and methods to implement the `Send` marker trait,
    /// ignoring type parameters.
    ///
    /// If a function is generic and its Future conditionally implements `Send`
    /// based on a generic parameter then it is considered `Send` and no warning is emitted.
    ///
    /// This can be used by library authors (public and internal) to ensure
    /// their functions are compatible with both multi-threaded runtimes that require `Send` futures,
    /// as well as single-threaded runtimes where callers may choose `!Send` types
    /// for generic parameters.
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
    /// ```no_run
    /// async fn not_send(bytes: std::rc::Rc<[u8]>) {}
    /// ```
    /// Use instead:
    /// ```no_run
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
        let ret_ty = return_ty(cx, cx.tcx.local_def_id_to_hir_id(fn_def_id).expect_owner());
        if let ty::Alias(ty::Opaque, AliasTy { def_id, args, .. }) = *ret_ty.kind()
            && let Some(future_trait) = cx.tcx.lang_items().future_trait()
            && let Some(send_trait) = cx.tcx.get_diagnostic_item(sym::Send)
        {
            let preds = cx.tcx.explicit_item_self_bounds(def_id);
            let is_future = preds.iter_instantiated_copied(cx.tcx, args).any(|(p, _)| {
                p.as_trait_clause()
                    .is_some_and(|trait_pred| trait_pred.skip_binder().trait_ref.def_id == future_trait)
            });
            if is_future {
                let span = decl.output.span();
                let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
                let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
                let cause = traits::ObligationCause::misc(span, fn_def_id);
                ocx.register_bound(cause, cx.param_env, ret_ty, send_trait);
                let send_errors = ocx.select_all_or_error();

                // Allow errors that try to prove `Send` for types that "mention" a generic parameter at the "top
                // level".
                // For example, allow errors that `T: Send` can't be proven, but reject `Rc<T>: Send` errors,
                // which is always unconditionally `!Send` for any possible type `T`.
                //
                // We also allow associated type projections if the self type is either itself a projection or a
                // type parameter.
                // This is to prevent emitting warnings for e.g. holding a `<Fut as Future>::Output` across await
                // points, where `Fut` is a type parameter.

                let is_send = send_errors.iter().all(|err| {
                    err.obligation
                        .predicate
                        .as_trait_clause()
                        .map(Binder::skip_binder)
                        .is_some_and(|pred| {
                            pred.def_id() == send_trait
                                && pred.self_ty().has_param()
                                && TyParamAtTopLevelVisitor.visit_ty(pred.self_ty()) == ControlFlow::Break(true)
                        })
                });

                if !is_send {
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

struct TyParamAtTopLevelVisitor;
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for TyParamAtTopLevelVisitor {
    type Result = ControlFlow<bool>;
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match ty.kind() {
            ty::Param(_) => ControlFlow::Break(true),
            ty::Alias(ty::AliasTyKind::Projection, ty) => ty.visit_with(self),
            _ => ControlFlow::Break(false),
        }
    }
}
