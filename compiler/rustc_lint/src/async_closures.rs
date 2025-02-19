use rustc_hir as hir;
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `closure_returning_async_block` lint detects cases where users
    /// write a closure that returns an async block.
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![warn(closure_returning_async_block)]
    /// let c = |x: &str| async {};
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Using an async closure is preferable over a closure that returns an
    /// async block, since async closures are less restrictive in how its
    /// captures are allowed to be used.
    ///
    /// For example, this code does not work with a closure returning an async
    /// block:
    ///
    /// ```rust,compile_fail
    /// async fn callback(x: &str) {}
    ///
    /// let captured_str = String::new();
    /// let c = move || async {
    ///     callback(&captured_str).await;
    /// };
    /// ```
    ///
    /// But it does work with async closures:
    ///
    /// ```rust
    /// async fn callback(x: &str) {}
    ///
    /// let captured_str = String::new();
    /// let c = async move || {
    ///     callback(&captured_str).await;
    /// };
    /// ```
    pub CLOSURE_RETURNING_ASYNC_BLOCK,
    Allow,
    "closure that returns `async {}` could be rewritten as an async closure",
}

declare_lint_pass!(
    /// Lint for potential usages of async closures and async fn trait bounds.
    AsyncClosureUsage => [CLOSURE_RETURNING_ASYNC_BLOCK]
);

impl<'tcx> LateLintPass<'tcx> for AsyncClosureUsage {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let hir::ExprKind::Closure(&hir::Closure {
            body,
            kind: hir::ClosureKind::Closure,
            fn_decl_span,
            ..
        }) = expr.kind
        else {
            return;
        };

        let mut body = cx.tcx.hir_body(body).value;

        // Only peel blocks that have no expressions.
        while let hir::ExprKind::Block(&hir::Block { stmts: [], expr: Some(tail), .. }, None) =
            body.kind
        {
            body = tail;
        }

        let hir::ExprKind::Closure(&hir::Closure {
            kind:
                hir::ClosureKind::Coroutine(hir::CoroutineKind::Desugared(
                    hir::CoroutineDesugaring::Async,
                    hir::CoroutineSource::Block,
                )),
            fn_decl_span: async_decl_span,
            ..
        }) = body.kind
        else {
            return;
        };

        let deletion_span = cx.tcx.sess.source_map().span_extend_while_whitespace(async_decl_span);

        cx.tcx.emit_node_span_lint(
            CLOSURE_RETURNING_ASYNC_BLOCK,
            expr.hir_id,
            fn_decl_span,
            ClosureReturningAsyncBlock {
                async_decl_span,
                sugg: AsyncClosureSugg {
                    deletion_span,
                    insertion_span: fn_decl_span.shrink_to_lo(),
                },
            },
        );
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_closure_returning_async_block)]
struct ClosureReturningAsyncBlock {
    #[label]
    async_decl_span: Span,
    #[subdiagnostic]
    sugg: AsyncClosureSugg,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "maybe-incorrect")]
struct AsyncClosureSugg {
    #[suggestion_part(code = "")]
    deletion_span: Span,
    #[suggestion_part(code = "async ")]
    insertion_span: Span,
}
