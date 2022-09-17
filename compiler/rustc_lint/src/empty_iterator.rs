use crate::{LateContext, LateLintPass, LintContext};

use rustc_ast::ast::LitKind;
use rustc_errors::fluent;
use rustc_hir::{Expr, ExprField, ExprKind, MatchSource, PatKind, StmtKind};
use rustc_span::Span;

declare_lint! {
    /// The `empty_iterator_range` lint checks for empty iterator ranges.
    ///
    /// ### Example
    ///
    /// ```rust
    /// for i in 10..0 { /* ... */ }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to have a statement that has no effect.
    pub EMPTY_ITERATOR_RANGE,
    Warn,
    "empty iterator range"
}

declare_lint_pass!(EmptyIteratorRange => [EMPTY_ITERATOR_RANGE]);

impl<'tcx> LateLintPass<'tcx> for EmptyIteratorRange {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let Some(for_expr) = extract_for_loop(expr) else { return };
        let span = expr.span.with_hi(for_expr.span.hi());
        if let ExprKind::Struct(_, expr_fields, _) = &for_expr.kind {
            is_empty_range(cx, span, expr_fields);
        } else if let ExprKind::MethodCall(_, expr, _, _) = &for_expr.kind {
            if let ExprKind::Struct(_, expr_fields, _) = &expr.kind {
                is_empty_range(cx, span, expr_fields);
            }
        }
    }
}

fn extract_for_loop<'tcx>(expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    if let ExprKind::DropTemps(e) = expr.kind
    && let ExprKind::Match(iterexpr, [arm], MatchSource::ForLoopDesugar) = e.kind
    && let ExprKind::Call(_, [arg]) = iterexpr.kind
    && let ExprKind::Loop(block, ..) = arm.body.kind
    && let [stmt] = block.stmts
    && let StmtKind::Expr(e) = stmt.kind
    && let ExprKind::Match(_, [_, some_arm], _) = e.kind
    && let PatKind::Struct(..) = some_arm.pat.kind
    {
        Some(arg)
    } else {
        None
    }
}

fn is_empty_range(cx: &LateContext<'_>, span: Span, expr_fields: &[ExprField<'_>]) {
    let mut prev = 0u128;
    for (index, expr_field) in expr_fields.iter().enumerate() {
        if let ExprKind::Lit(lit) = &expr_field.expr.kind {
            if let LitKind::Int(u, _) = lit.node {
                if index == 0 {
                    prev = u;
                } else if prev > u {
                    cx.struct_span_lint(
                        EMPTY_ITERATOR_RANGE,
                        span,
                        fluent::lint::empty_iter_ranges,
                        |lint| {
                            lint.span_label(span, fluent::lint::clarification)
                                .note(fluent::lint::note)
                                .help(fluent::lint::help)
                        },
                    );
                }
            }
        }
    }
}
