use clippy_utils::diagnostics::{span_lint_and_then, span_lint_hir};
use clippy_utils::get_parent_expr;
use clippy_utils::sugg::Sugg;
use hir::Param;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{
    CaptureBy, ClosureKind, CoroutineDesugaring, CoroutineKind, CoroutineSource, ExprKind, intravisit as hir_visit,
};
use rustc_lint::{LateContext, LateLintPass, LintContext as _};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use rustc_span::ExpnKind;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Detects closures called in the same expression where they
    /// are defined.
    ///
    /// ### Why is this bad?
    /// It is unnecessarily adding to the expression's
    /// complexity.
    ///
    /// ### Example
    /// ```no_run
    /// let a = (|| 42)();
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let a = 42;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub REDUNDANT_CLOSURE_CALL,
    complexity,
    "throwaway closures called in the expression they are defined"
}

declare_lint_pass!(RedundantClosureCall => [REDUNDANT_CLOSURE_CALL]);

// Used to find `return` statements or equivalents e.g., `?`
struct ReturnVisitor;

impl<'tcx> Visitor<'tcx> for ReturnVisitor {
    type Result = ControlFlow<()>;
    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) -> ControlFlow<()> {
        if let ExprKind::Ret(_) | ExprKind::Match(.., hir::MatchSource::TryDesugar(_)) = ex.kind {
            return ControlFlow::Break(());
        }
        hir_visit::walk_expr(self, ex)
    }
}

/// Checks if the expression contains a return statement of any kind using `ReturnVisitor`.
fn contains_early_return<'tcx>(expr: &'tcx hir::Expr<'tcx>) -> bool {
    let mut visitor = ReturnVisitor;
    visitor.visit_expr(expr).is_break()
}

/// Matches the inner closure of an async closure like `async || { 42 }`.
const DESUGARED_ASYNC_CLOSURE_KIND: ClosureKind = ClosureKind::Coroutine(CoroutineKind::Desugared(
    CoroutineDesugaring::Async,
    CoroutineSource::Closure,
));

/// Matches the inner closure of a closure returning an async block like `|| async { 42 }`.
const DESUGARED_ASYNC_BLOCK_CLOSURE_KIND: ClosureKind = ClosureKind::Coroutine(CoroutineKind::Desugared(
    CoroutineDesugaring::Async,
    CoroutineSource::Block,
));

/// Tries to find the innermost closure:
/// ```rust,ignore
/// (|| || || || 42)()()()()
///  ^^^^^^^^^^^^^^          given this nested closure expression
///           ^^^^^          we want to return this closure
/// ```
/// It also has a parameter for how many steps to go in at most, so as to
/// not take more closures than there are calls.
fn find_innermost_closure<'tcx>(
    cx: &LateContext<'tcx>,
    mut expr: &'tcx hir::Expr<'tcx>,
    mut steps: usize,
) -> Option<(
    &'tcx hir::Expr<'tcx>,
    &'tcx hir::FnDecl<'tcx>,
    ty::Asyncness,
    &'tcx [Param<'tcx>],
    CaptureBy,
)> {
    let mut data = None;

    while let ExprKind::Closure(closure) = expr.kind
        && let body = cx.tcx.hir_body(closure.body)
        && !contains_early_return(body.value)
        && steps > 0
    {
        let mut unwrapped_body_value = body.value;
        let mut asyncness = ty::Asyncness::No;
        let mut capture_clause = closure.capture_clause;

        if let ExprKind::Closure(inner_closure) = body.value.kind {
            if matches!(inner_closure.kind, DESUGARED_ASYNC_CLOSURE_KIND) {
                asyncness = ty::Asyncness::Yes;
                unwrapped_body_value = cx.tcx.hir_body(inner_closure.body).value;
            } else if matches!(inner_closure.kind, DESUGARED_ASYNC_BLOCK_CLOSURE_KIND) {
                asyncness = ty::Asyncness::Yes;
                capture_clause = inner_closure.capture_clause;
                unwrapped_body_value = cx.tcx.hir_body(inner_closure.body).value;
            }

            if contains_early_return(unwrapped_body_value) {
                break;
            }
        }

        expr = body.value;
        data = Some((
            unwrapped_body_value,
            closure.fn_decl,
            asyncness,
            body.params,
            capture_clause,
        ));
        steps -= 1;
    }

    data
}

/// Returns the capture keyword for a closure. One of:
/// `move`, `use`, or nothing for capture by reference.
fn capture_keyword(capture_clause: CaptureBy) -> &'static str {
    match capture_clause {
        CaptureBy::Value { .. } => "move ",
        CaptureBy::Use { .. } => "use ",
        CaptureBy::Ref => "",
    }
}

/// "Walks up" the chain of calls to find the outermost call expression, and returns the depth:
/// ```rust,ignore
/// (|| || || 3)()()()
///             ^^      this is the call expression we were given
///                 ^^  this is what we want to return (and the depth is 3)
/// ```
fn get_parent_call_exprs<'tcx>(
    cx: &LateContext<'tcx>,
    mut expr: &'tcx hir::Expr<'tcx>,
) -> (&'tcx hir::Expr<'tcx>, usize) {
    let mut depth = 1;
    while let Some(parent) = get_parent_expr(cx, expr)
        && let ExprKind::Call(recv, _) = parent.kind
        && expr.span == recv.span
    {
        expr = parent;
        depth += 1;
    }
    (expr, depth)
}

impl<'tcx> LateLintPass<'tcx> for RedundantClosureCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        if let ExprKind::Call(recv, _) = expr.kind
            // don't lint if the receiver is a call, too.
            // we do this in order to prevent linting multiple times; consider:
            // `(|| || 1)()()`
            //           ^^  we only want to lint for this call (but we walk up the calls to consider both calls).
            // without this check, we'd end up linting twice.
            && !matches!(recv.kind, ExprKind::Call(..))
            // Check if `recv` comes from a macro expansion. If it does, make sure that it's an expansion that is
            // the same as the one the call is in.
            // For instance, let's assume `x!()` returns a closure:
            //    B ---v
            //      x!()()
            //          ^- A
            // The call happens in the expansion `A`, while the closure originates from the expansion `B`.
            // We don't want to suggest replacing `x!()()` with `x!()`.
            && recv.span.ctxt().outer_expn() == expr.span.ctxt().outer_expn()
            && let (full_expr, call_depth) = get_parent_call_exprs(cx, expr)
            && let Some((mut body, fn_decl, coroutine_kind, params, capture_clause)) =
                find_innermost_closure(cx, recv, call_depth)
            // outside macros we lint properly. Inside macros, we lint only ||() style closures.
            && (!matches!(expr.span.ctxt().outer_expn_data().kind, ExpnKind::Macro(_, _)) || params.is_empty())
        {
            span_lint_and_then(
                cx,
                REDUNDANT_CLOSURE_CALL,
                full_expr.span,
                "try not to call a closure in the expression where it is declared",
                |diag| {
                    if fn_decl.inputs.is_empty() {
                        let mut applicability = Applicability::MachineApplicable;
                        let mut hint =
                            Sugg::hir_with_context(cx, body, full_expr.span.ctxt(), "..", &mut applicability);

                        if coroutine_kind.is_async() {
                            if let ExprKind::Closure(closure) = body.kind {
                                // Like `async fn`, async closures are wrapped in an additional block
                                // to move all of the closure's arguments into the future.
                                body = cx.tcx.hir_body(closure.body).value;
                            }

                            // `async x` is a syntax error, so it becomes `async { x }`
                            if let ExprKind::Block(block, _) = body.kind
                                && let Some(block_expr) = block.expr
                                && let ExprKind::DropTemps(body_expr) = block_expr.kind
                                && !matches!(body_expr.kind, ExprKind::Block(_, _))
                            {
                                hint = hint.blockify();
                            }

                            hint = Sugg::NonParen(format!("async {}{hint}", capture_keyword(capture_clause)).into());
                        } else if let ExprKind::Block(block, _) = body.kind
                            && block.stmts.is_empty()
                            && let Some(expr) = block.expr
                        {
                            // If the (non-async) closure body is a block with a single expression,
                            // suggest just the inner expression, not the block.
                            // Example: `(|| { Some(true) })()` should suggest `Some(true)`
                            hint = Sugg::hir_with_context(cx, expr, full_expr.span.ctxt(), "..", &mut applicability)
                                .maybe_paren();
                        }

                        diag.span_suggestion(full_expr.span, "try doing something like", hint, applicability);
                    }
                },
            );
        }
    }

    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx hir::Block<'_>) {
        fn count_closure_usage<'tcx>(
            cx: &LateContext<'tcx>,
            block: &'tcx hir::Block<'_>,
            path: &'tcx hir::Path<'tcx>,
        ) -> usize {
            struct ClosureUsageCount<'a, 'tcx> {
                cx: &'a LateContext<'tcx>,
                path: &'tcx hir::Path<'tcx>,
                count: usize,
            }
            impl<'tcx> Visitor<'tcx> for ClosureUsageCount<'_, 'tcx> {
                type NestedFilter = nested_filter::OnlyBodies;

                fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
                    if let ExprKind::Call(closure, _) = expr.kind
                        && let ExprKind::Path(hir::QPath::Resolved(_, path)) = closure.kind
                        && self.path.segments[0].ident == path.segments[0].ident
                        && self.path.res == path.res
                    {
                        self.count += 1;
                    }
                    hir_visit::walk_expr(self, expr);
                }

                fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
                    self.cx.tcx
                }
            }
            let mut closure_usage_count = ClosureUsageCount { cx, path, count: 0 };
            closure_usage_count.visit_block(block);
            closure_usage_count.count
        }

        for w in block.stmts.windows(2) {
            if let hir::StmtKind::Let(local) = w[0].kind
                && let Some(t) = local.init
                && let ExprKind::Closure { .. } = t.kind
                && let hir::PatKind::Binding(_, _, ident, _) = local.pat.kind
                && let hir::StmtKind::Semi(second) = w[1].kind
                && let ExprKind::Assign(_, call, _) = second.kind
                && let ExprKind::Call(closure, _) = call.kind
                && let ExprKind::Path(hir::QPath::Resolved(_, path)) = closure.kind
                && ident == path.segments[0].ident
                && count_closure_usage(cx, block, path) == 1
            {
                span_lint_hir(
                    cx,
                    REDUNDANT_CLOSURE_CALL,
                    second.hir_id,
                    second.span,
                    "closure called just once immediately after it was declared",
                );
            }
        }
    }
}
