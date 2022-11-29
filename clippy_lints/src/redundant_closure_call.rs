use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::sugg::Sugg;
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_ast::visit as ast_visit;
use rustc_ast::visit::Visitor as AstVisitor;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit as hir_visit;
use rustc_hir::intravisit::Visitor as HirVisitor;
use rustc_lint::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
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

impl<'ast> ast_visit::Visitor<'ast> for ReturnVisitor {
    fn visit_expr(&mut self, ex: &'ast ast::Expr) {
        if let ast::ExprKind::Ret(_) | ast::ExprKind::Try(_) = ex.kind {
            self.found_return = true;
        }

        ast_visit::walk_expr(self, ex);
    }
}

impl EarlyLintPass for RedundantClosureCall {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }
        if_chain! {
            if let ast::ExprKind::Call(ref paren, _) = expr.kind;
            if let ast::ExprKind::Paren(ref closure) = paren.kind;
            if let ast::ExprKind::Closure(box ast::Closure { ref asyncness, ref fn_decl, ref body, .. }) = closure.kind;
            then {
                let mut visitor = ReturnVisitor::new();
                visitor.visit_expr(body);
                if !visitor.found_return {
                    span_lint_and_then(
                        cx,
                        REDUNDANT_CLOSURE_CALL,
                        expr.span,
                        "try not to call a closure in the expression where it is declared",
                        |diag| {
                            if fn_decl.inputs.is_empty() {
                                let mut app = Applicability::MachineApplicable;
                                let mut hint = Sugg::ast(cx, body, "..", closure.span.ctxt(), &mut app);

                                if asyncness.is_async() {
                                    // `async x` is a syntax error, so it becomes `async { x }`
                                    if !matches!(body.kind, ast::ExprKind::Block(_, _)) {
                                        hint = hint.blockify();
                                    }

                                    hint = hint.asyncify();
                                }

                                diag.span_suggestion(expr.span, "try doing something like", hint.to_string(), app);
                            }
                        },
                    );
                }
            }
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for RedundantClosureCall {
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
