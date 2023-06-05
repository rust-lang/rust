use clippy_utils::source::snippet;
use clippy_utils::{diagnostics::span_lint_and_sugg, ty::implements_trait};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_target::abi::Size;

declare_clippy_lint! {
    /// ### What it does
    /// It checks for the size of a `Future` created by `async fn` or `async {}`.
    ///
    /// ### Why is this bad?
    /// Due to the current [unideal implementation](https://github.com/rust-lang/rust/issues/69826) of `Generator`,
    /// large size of a `Future` may cause stack overflows.
    ///
    /// ### Example
    /// ```rust
    /// async fn wait(f: impl std::future::Future<Output = ()>) {}
    ///
    /// async fn big_fut(arg: [u8; 1024]) {}
    ///
    /// pub async fn test() {
    ///     let fut = big_fut([0u8; 1024]);
    ///     wait(fut).await;
    /// }
    /// ```
    ///
    /// `Box::pin` the big future instead.
    ///
    /// ```rust
    /// async fn wait(f: impl std::future::Future<Output = ()>) {}
    ///
    /// async fn big_fut(arg: [u8; 1024]) {}
    ///
    /// pub async fn test() {
    ///     let fut = Box::pin(big_fut([0u8; 1024]));
    ///     wait(fut).await;
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub LARGE_FUTURES,
    pedantic,
    "large future may lead to unexpected stack overflows"
}

#[derive(Copy, Clone)]
pub struct LargeFuture {
    future_size_threshold: u64,
}

impl LargeFuture {
    pub fn new(future_size_threshold: u64) -> Self {
        Self { future_size_threshold }
    }
}

impl_lint_pass!(LargeFuture => [LARGE_FUTURES]);

impl<'tcx> LateLintPass<'tcx> for LargeFuture {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if matches!(expr.span.ctxt().outer_expn_data().kind, rustc_span::ExpnKind::Macro(..)) {
            return;
        }
        if let ExprKind::Match(expr, _, MatchSource::AwaitDesugar) = expr.kind {
            if let ExprKind::Call(func, [expr, ..]) = expr.kind
                && let ExprKind::Path(QPath::LangItem(LangItem::IntoFutureIntoFuture, ..)) = func.kind
                && let ty = cx.typeck_results().expr_ty(expr)
                && let Some(future_trait_def_id) = cx.tcx.lang_items().future_trait()
                && implements_trait(cx, ty, future_trait_def_id, &[])
                && let Ok(layout) = cx.tcx.layout_of(cx.param_env.and(ty))
                && let size = layout.layout.size()
                && size >= Size::from_bytes(self.future_size_threshold)
            {
                span_lint_and_sugg(
                    cx,
                    LARGE_FUTURES,
                    expr.span,
                    &format!("large future with a size of {} bytes", size.bytes()),
                    "consider `Box::pin` on it",
                    format!("Box::pin({})", snippet(cx, expr.span, "..")),
                    Applicability::Unspecified,
                );
            }
        }
    }
}
