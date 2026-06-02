use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::implements_trait;
use clippy_utils::{desugar_await, desugared_async_block, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::UpvarCapture;
use rustc_session::declare_lint_pass;
use rustc_span::{ExpnKind, Span};

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
        if !expr.span.in_external_macro(cx.tcx.sess.source_map()) &&
            let Some(body_expr) = desugar_async_block(cx, expr) &&
            let Some(inner_expr) = desugar_await(peel_blocks(body_expr)) &&
            // The await prefix must not come from a macro as its content could change in the future.
            !is_from_macro_within(inner_expr.span, body_expr.span) &&
            // The await prefix must implement Future, as implementing IntoFuture is not enough.
            let Some(future_trait) = cx.tcx.lang_items().future_trait() &&
            implements_trait(cx, cx.typeck_results().expr_ty(inner_expr), future_trait, &[]) &&
            // An async block does not have immediate side-effects from a `.await` point-of-view.
            (!inner_expr.can_have_side_effects() || desugar_async_block(cx, inner_expr).is_some())
        {
            span_lint_and_sugg(
                cx,
                REDUNDANT_ASYNC_BLOCK,
                expr.span,
                "this async expression only awaits a single future",
                "you can reduce it to",
                snippet(cx, inner_expr.span, "..").into_owned(),
                Applicability::MachineApplicable,
            );
        }
    }
}

/// If `expr` is a desugared `async` block, return the original expression if it does not capture
/// any variable by ref.
fn desugar_async_block<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Expr<'tcx>> {
    let (def_id, body) = desugared_async_block(cx, expr)?;
    if cx.typeck_results()
            .closure_min_captures
            .get(&def_id)
            .is_none_or(|m| {
                m.values().all(|places| {
                    places
                        .iter()
                        .all(|place| matches!(place.info.capture_kind, UpvarCapture::ByValue))
                })
            })
    {
        Some(body.value)
    } else {
        None
    }
}

fn is_from_macro_within(mut span: Span, outer_span: Span) -> bool {
    let outer_ctxt = outer_span.ctxt();
    loop {
        let ctxt = span.ctxt();
        if ctxt.is_root() || ctxt == outer_ctxt {
            break
        }

        let expn_data = ctxt.outer_expn_data();
        match expn_data.kind {
            ExpnKind::Macro { .. } => return true,
            ExpnKind::Root
            | ExpnKind::AstPass(_)
            | ExpnKind::Desugaring(_) => {}
        }
        span = expn_data.call_site;
    }

    false
}
