use crate::rustc_lint::LintContext;
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::get_parent_expr;
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit as hir_visit;
use rustc_hir::intravisit::{Visitor as HirVisitor, Visitor};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
    /// ```rust
    /// let a = (|| 42)();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let a = 42;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub REDUNDANT_CLOSURE_CALL,
    complexity,
    "throwaway closures called in the expression they are defined"
}

declare_lint_pass!(RedundantClosureCall => [REDUNDANT_CLOSURE_CALL]);

// Used to find `return` statements or equivalents e.g., `?`
struct ReturnVisitor {
    found_return: bool,
}

impl ReturnVisitor {
    #[must_use]
    fn new() -> Self {
        Self { found_return: false }
    }
}

impl<'tcx> Visitor<'tcx> for ReturnVisitor {
    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Ret(_) | hir::ExprKind::Match(.., hir::MatchSource::TryDesugar) = ex.kind {
            self.found_return = true;
        } else {
            hir_visit::walk_expr(self, ex);
        }
    }
}

/// Checks if the body is owned by an async closure
fn is_async_closure(body: &hir::Body<'_>) -> bool {
    if let hir::ExprKind::Closure(closure) = body.value.kind
        && let [resume_ty] = closure.fn_decl.inputs
        && let hir::TyKind::Path(hir::QPath::LangItem(hir::LangItem::ResumeTy, ..)) = resume_ty.kind
    {
        true
    } else {
        false
    }
}

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
) -> Option<(&'tcx hir::Expr<'tcx>, &'tcx hir::FnDecl<'tcx>, hir::IsAsync)> {
    let mut data = None;

    while let hir::ExprKind::Closure(closure) = expr.kind
        && let body = cx.tcx.hir().body(closure.body)
        && {
            let mut visitor = ReturnVisitor::new();
            visitor.visit_expr(body.value);
            !visitor.found_return
        }
        && steps > 0
    {
        expr = body.value;
        data = Some((body.value, closure.fn_decl, if is_async_closure(body) {
            hir::IsAsync::Async
        } else {
            hir::IsAsync::NotAsync
        }));
        steps -= 1;
    }

    data
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
        && let hir::ExprKind::Call(recv, _) = parent.kind
        && expr.span == recv.span
    {
        expr = parent;
        depth += 1;
    }
    (expr, depth)
}

impl<'tcx> LateLintPass<'tcx> for RedundantClosureCall {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let hir::ExprKind::Call(recv, _) = expr.kind
            // don't lint if the receiver is a call, too. 
            // we do this in order to prevent linting multiple times; consider:
            // `(|| || 1)()()`
            //           ^^  we only want to lint for this call (but we walk up the calls to consider both calls).
            // without this check, we'd end up linting twice.
            && !matches!(recv.kind, hir::ExprKind::Call(..))
            && let (full_expr, call_depth) = get_parent_call_exprs(cx, expr)
            && let Some((body, fn_decl, generator_kind)) = find_innermost_closure(cx, recv, call_depth)
        {
            span_lint_and_then(
                cx,
                REDUNDANT_CLOSURE_CALL,
                full_expr.span,
                "try not to call a closure in the expression where it is declared",
                |diag| {
                    if fn_decl.inputs.is_empty() {
                        let mut applicability = Applicability::MachineApplicable;
                        let mut hint = Sugg::hir_with_context(cx, body, full_expr.span.ctxt(), "..", &mut applicability);

                        if generator_kind.is_async()
                            && let hir::ExprKind::Closure(closure) = body.kind
                        {
                            let async_closure_body = cx.tcx.hir().body(closure.body);

                            // `async x` is a syntax error, so it becomes `async { x }`
                            if !matches!(async_closure_body.value.kind, hir::ExprKind::Block(_, _)) {
                                hint = hint.blockify();
                            }

                            hint = hint.asyncify();
                        }

                        diag.span_suggestion(
                            full_expr.span,
                            "try doing something like",
                            hint.maybe_par(),
                            applicability
                        );
                    }
                }
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
            impl<'a, 'tcx> hir_visit::Visitor<'tcx> for ClosureUsageCount<'a, 'tcx> {
                type NestedFilter = nested_filter::OnlyBodies;

                fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
                    if_chain! {
                        if let hir::ExprKind::Call(closure, _) = expr.kind;
                        if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = closure.kind;
                        if self.path.segments[0].ident == path.segments[0].ident;
                        if self.path.res == path.res;
                        then {
                            self.count += 1;
                        }
                    }
                    hir_visit::walk_expr(self, expr);
                }

                fn nested_visit_map(&mut self) -> Self::Map {
                    self.cx.tcx.hir()
                }
            }
            let mut closure_usage_count = ClosureUsageCount { cx, path, count: 0 };
            closure_usage_count.visit_block(block);
            closure_usage_count.count
        }

        for w in block.stmts.windows(2) {
            if_chain! {
                if let hir::StmtKind::Local(local) = w[0].kind;
                if let Option::Some(t) = local.init;
                if let hir::ExprKind::Closure { .. } = t.kind;
                if let hir::PatKind::Binding(_, _, ident, _) = local.pat.kind;
                if let hir::StmtKind::Semi(second) = w[1].kind;
                if let hir::ExprKind::Assign(_, call, _) = second.kind;
                if let hir::ExprKind::Call(closure, _) = call.kind;
                if let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = closure.kind;
                if ident == path.segments[0].ident;
                if count_closure_usage(cx, block, path) == 1;
                then {
                    span_lint(
                        cx,
                        REDUNDANT_CLOSURE_CALL,
                        second.span,
                        "closure called just once immediately after it was declared",
                    );
                }
            }
        }
    }
}
