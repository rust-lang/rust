use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{snippet, walk_span_to_context};
use clippy_utils::ty::implements_trait;
use clippy_utils::{desugar_await, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Closure, ClosureKind, CoroutineDesugaring, CoroutineKind, CoroutineSource, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::UpvarCapture;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `async` block that only returns `await` on a future.
    ///
    /// ### Why is this bad?
    /// It is simpler and more efficient to use the future directly.
    ///
    /// ### Example
    /// ```no_run
    /// let f = async {
    ///     1 + 2
    /// };
    /// let fut = async {
    ///     f.await
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// let f = async {
    ///     1 + 2
    /// };
    /// let fut = f;
    /// ```
    #[clippy::version = "1.70.0"]
    pub REDUNDANT_ASYNC_BLOCK,
    complexity,
    "`async { future.await }` can be replaced by `future`"
}
declare_lint_pass!(RedundantAsyncBlock => [REDUNDANT_ASYNC_BLOCK]);

impl<'tcx> LateLintPass<'tcx> for RedundantAsyncBlock {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let span = expr.span;
        if !span.in_external_macro(cx.tcx.sess.source_map()) &&
            let Some(body_expr) = desugar_async_block(cx, expr) &&
            let Some(expr) = desugar_await(peel_blocks(body_expr)) &&
            // The await prefix must not come from a macro as its content could change in the future.
            expr.span.eq_ctxt(body_expr.span) &&
            // The await prefix must implement Future, as implementing IntoFuture is not enough.
            let Some(future_trait) = cx.tcx.lang_items().future_trait() &&
            implements_trait(cx, cx.typeck_results().expr_ty(expr), future_trait, &[]) &&
            // An async block does not have immediate side-effects from a `.await` point-of-view.
            (!expr.can_have_side_effects() || desugar_async_block(cx, expr).is_some()) &&
            let Some(shortened_span) = walk_span_to_context(expr.span, span.ctxt())
        {
            span_lint_and_sugg(
                cx,
                REDUNDANT_ASYNC_BLOCK,
                span,
                "this async expression only awaits a single future",
                "you can reduce it to",
                snippet(cx, shortened_span, "..").into_owned(),
                Applicability::MachineApplicable,
            );
        }
    }
}

/// If `expr` is a desugared `async` block, return the original expression if it does not capture
/// any variable by ref.
fn desugar_async_block<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::Closure(Closure { body, def_id, kind, .. }) = expr.kind
        && let body = cx.tcx.hir_body(*body)
        && matches!(
            kind,
            ClosureKind::Coroutine(CoroutineKind::Desugared(
                CoroutineDesugaring::Async,
                CoroutineSource::Block
            ))
        )
    {
        cx.typeck_results()
            .closure_min_captures
            .get(def_id)
            .is_none_or(|m| {
                m.values().all(|places| {
                    places
                        .iter()
                        .all(|place| matches!(place.info.capture_kind, UpvarCapture::ByValue))
                })
            })
            .then_some(body.value)
    } else {
        None
    }
}
