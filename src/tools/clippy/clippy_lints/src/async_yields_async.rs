use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::implements_trait;
use rustc_errors::Applicability;
use rustc_hir::{Closure, ClosureKind, CoroutineDesugaring, CoroutineKind, CoroutineSource, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for async blocks that yield values of types
    /// that can themselves be awaited.
    ///
    /// ### Why is this bad?
    /// An await is likely missing.
    ///
    /// ### Example
    /// ```no_run
    /// async fn foo() {}
    ///
    /// fn bar() {
    ///   let x = async {
    ///     foo()
    ///   };
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// async fn foo() {}
    ///
    /// fn bar() {
    ///   let x = async {
    ///     foo().await
    ///   };
    /// }
    /// ```
    #[clippy::version = "1.48.0"]
    pub ASYNC_YIELDS_ASYNC,
    correctness,
    "async blocks that return a type that can be awaited"
}

declare_lint_pass!(AsyncYieldsAsync => [ASYNC_YIELDS_ASYNC]);

impl<'tcx> LateLintPass<'tcx> for AsyncYieldsAsync {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let ExprKind::Closure(Closure {
            kind: ClosureKind::Coroutine(CoroutineKind::Desugared(CoroutineDesugaring::Async, kind)),
            body: body_id,
            ..
        }) = expr.kind
        else {
            return;
        };

        let body_expr = match kind {
            CoroutineSource::Fn => {
                // For functions, with explicitly defined types, don't warn.
                // XXXkhuey maybe we should?
                return;
            },
            CoroutineSource::Block => cx.tcx.hir().body(*body_id).value,
            CoroutineSource::Closure => {
                // Like `async fn`, async closures are wrapped in an additional block
                // to move all of the closure's arguments into the future.

                let async_closure_body = cx.tcx.hir().body(*body_id).value;
                let ExprKind::Block(block, _) = async_closure_body.kind else {
                    return;
                };
                let Some(block_expr) = block.expr else {
                    return;
                };
                let ExprKind::DropTemps(body_expr) = block_expr.kind else {
                    return;
                };
                body_expr
            },
        };

        let Some(future_trait_def_id) = cx.tcx.lang_items().future_trait() else {
            return;
        };

        let typeck_results = cx.tcx.typeck_body(*body_id);
        let expr_ty = typeck_results.expr_ty(body_expr);

        if implements_trait(cx, expr_ty, future_trait_def_id, &[]) {
            let return_expr_span = match &body_expr.kind {
                // XXXkhuey there has to be a better way.
                ExprKind::Block(block, _) => block.expr.map(|e| e.span),
                ExprKind::Path(QPath::Resolved(_, path)) => Some(path.span),
                _ => None,
            };
            if let Some(return_expr_span) = return_expr_span {
                span_lint_hir_and_then(
                    cx,
                    ASYNC_YIELDS_ASYNC,
                    body_expr.hir_id,
                    return_expr_span,
                    "an async construct yields a type which is itself awaitable",
                    |db| {
                        db.span_label(body_expr.span, "outer async construct");
                        db.span_label(return_expr_span, "awaitable value not awaited");
                        db.span_suggestion(
                            return_expr_span,
                            "consider awaiting this value",
                            format!("{}.await", snippet(cx, return_expr_span, "..")),
                            Applicability::MaybeIncorrect,
                        );
                    },
                );
            }
        }
    }
}
