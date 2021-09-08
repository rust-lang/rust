use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::intravisit::{walk_expr, walk_fn, FnKind, NestedVisitorMap, Visitor};
use rustc_hir::{Body, Expr, ExprKind, FnDecl, FnHeader, HirId, IsAsync, YieldSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that are declared `async` but have no `.await`s inside of them.
    ///
    /// ### Why is this bad?
    /// Async functions with no async code create overhead, both mentally and computationally.
    /// Callers of async methods either need to be calling from an async function themselves or run it on an executor, both of which
    /// causes runtime overhead and hassle for the caller.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// async fn get_random_number() -> i64 {
    ///     4 // Chosen by fair dice roll. Guaranteed to be random.
    /// }
    /// let number_future = get_random_number();
    ///
    /// // Good
    /// fn get_random_number_improved() -> i64 {
    ///     4 // Chosen by fair dice roll. Guaranteed to be random.
    /// }
    /// let number_future = async { get_random_number_improved() };
    /// ```
    pub UNUSED_ASYNC,
    pedantic,
    "finds async functions with no await statements"
}

declare_lint_pass!(UnusedAsync => [UNUSED_ASYNC]);

struct AsyncFnVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    found_await: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for AsyncFnVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if let ExprKind::Yield(_, YieldSource::Await { .. }) = ex.kind {
            self.found_await = true;
        }
        walk_expr(self, ex);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }
}

impl<'tcx> LateLintPass<'tcx> for UnusedAsync {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        fn_decl: &'tcx FnDecl<'tcx>,
        body: &Body<'tcx>,
        span: Span,
        hir_id: HirId,
    ) {
        if let FnKind::ItemFn(_, _, FnHeader { asyncness, .. }, _) = &fn_kind {
            if matches!(asyncness, IsAsync::Async) {
                let mut visitor = AsyncFnVisitor { cx, found_await: false };
                walk_fn(&mut visitor, fn_kind, fn_decl, body.id(), span, hir_id);
                if !visitor.found_await {
                    span_lint_and_help(
                        cx,
                        UNUSED_ASYNC,
                        span,
                        "unused `async` for function with no await statements",
                        None,
                        "consider removing the `async` from this function",
                    );
                }
            }
        }
    }
}
