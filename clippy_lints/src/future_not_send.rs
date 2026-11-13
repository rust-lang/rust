use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_lint_allowed, ty_is_future};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl};
use rustc_infer::infer::TyCtxtInferExt as _;
use rustc_lint::{LateContext, LateLintPass, declare_lint_pass};
use rustc_middle::ty::print::PrintTraitRefExt as _;
use rustc_middle::ty::{
    self, AliasTy, Binder, ClauseKind, PredicateKind, Ty, TyCtxt, TypeVisitable as _, TypeVisitableExt as _,
    TypeVisitor,
};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt as _;
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
        // FIXME: We could potentially replace "clippy_utils::return_ty" with this,
        // but doing so causes regressions in some lints that rely on structural
        // equality of types that haven't been fully normalized. Keep this separate for
        // now and revisit the possibility later.
        fn normalized_return_ty<'tcx>(cx: &LateContext<'tcx>, fn_def_id: rustc_hir::OwnerId) -> Option<Ty<'tcx>> {
            debug_assert!(cx.tcx.hir_owner_node(fn_def_id).fn_sig().is_some());

            let ret_ty = cx
                .tcx
                .fn_sig(fn_def_id)
                .instantiate_identity()
                .map(Binder::<'_, ty::FnSig<'_>>::output);
            let ret_ty = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), ret_ty).ok()?;
            Some(cx.tcx.instantiate_bound_regions_with_erased(ret_ty))
        }

        let hir_id = cx.tcx.local_def_id_to_hir_id(fn_def_id);
        // Proving `Send` for large, deeply nested futures can be very expensive, if this lint is only enabled for parts
        // of a crate. Therefore check if the lint is enabled here before proceeding. See `matrix-sdk` for benchmarking
        // example :(
        if matches!(kind, FnKind::Closure) || is_lint_allowed(cx, FUTURE_NOT_SEND, hir_id) {
            return;
        }

        if let Some(ret_ty) = normalized_return_ty(cx, hir_id.expect_owner())
            && let Some(send_trait) = cx.tcx.get_diagnostic_item(sym::Send)
            && ty_is_future(cx, ret_ty)
        {
            let span = decl.output.span();
            let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
            let ocx = ObligationCtxt::new_with_diagnostics(&infcx);
            let cause = traits::ObligationCause::misc(span, fn_def_id);
            ocx.register_bound(cause, cx.param_env, ret_ty, send_trait);
            let send_errors = ocx.evaluate_obligations_error_on_ambiguity();

            // Allow errors that try to prove `Send` for types that "mention" a generic parameter at the "top
            // level".
            // For example, allow errors that `T: Send` can't be proven, but reject `Rc<T>: Send` errors,
            // which is always unconditionally `!Send` for any possible type `T`.
            //
            // We also allow associated type projections if the self type is either itself a projection or a
            // type parameter.
            // This is to prevent emitting warnings for e.g. holding a `<Fut as Future>::Output` across await
            // points, where `Fut` is a type parameter.

            let is_send = send_errors.as_slice().iter().all(|err| {
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

struct TyParamAtTopLevelVisitor;
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for TyParamAtTopLevelVisitor {
    type Result = ControlFlow<bool>;
    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match ty.kind() {
            ty::Param(_) => ControlFlow::Break(true),
            ty::Alias(
                _,
                ty @ AliasTy {
                    kind: ty::Projection { .. },
                    ..
                },
            ) => ty.visit_with(self),
            _ => ControlFlow::Break(false),
        }
    }
}
