use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::implements_trait;
use rustc_abi::Size;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// It checks for the size of a `Future` created by `async fn` or `async {}`.
    ///
    /// ### Why is this bad?
    /// Due to the current [unideal implementation](https://github.com/rust-lang/rust/issues/69826) of `Coroutine`,
    /// large size of a `Future` may cause stack overflows.
    ///
    /// ### Example
    /// ```no_run
    /// async fn large_future(_x: [u8; 16 * 1024]) {}
    ///
    /// pub async fn trigger() {
    ///     large_future([0u8; 16 * 1024]).await;
    /// }
    /// ```
    ///
    /// `Box::pin` the big future instead.
    ///
    /// ```no_run
    /// async fn large_future(_x: [u8; 16 * 1024]) {}
    ///
    /// pub async fn trigger() {
    ///     Box::pin(large_future([0u8; 16 * 1024])).await;
    /// }
    /// ```
    #[clippy::version = "1.70.0"]
    pub LARGE_FUTURES,
    pedantic,
    "large future may lead to unexpected stack overflows"
}

pub struct LargeFuture {
    future_size_threshold: u64,
}

impl LargeFuture {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            future_size_threshold: conf.future_size_threshold,
        }
    }
}

impl_lint_pass!(LargeFuture => [LARGE_FUTURES]);

impl<'tcx> LateLintPass<'tcx> for LargeFuture {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Match(scrutinee, _, MatchSource::AwaitDesugar) = expr.kind
            && let ExprKind::Call(func, [arg]) = scrutinee.kind
            && let ExprKind::Path(QPath::LangItem(LangItem::IntoFutureIntoFuture, ..)) = func.kind
            && !expr.span.from_expansion()
            && let ty = cx.typeck_results().expr_ty(arg)
            && let Some(future_trait_def_id) = cx.tcx.lang_items().future_trait()
            && implements_trait(cx, ty, future_trait_def_id, &[])
            && let Ok(layout) = cx.tcx.layout_of(cx.typing_env().as_query_input(ty))
            && let size = layout.layout.size()
            && size >= Size::from_bytes(self.future_size_threshold)
        {
            span_lint_and_sugg(
                cx,
                LARGE_FUTURES,
                arg.span,
                format!("large future with a size of {} bytes", size.bytes()),
                "consider `Box::pin` on it",
                format!("Box::pin({})", snippet(cx, arg.span, "..")),
                Applicability::Unspecified,
            );
        }
    }
}
