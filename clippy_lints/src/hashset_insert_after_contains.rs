use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{higher, peel_hir_expr_while, SpanlessEq};
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `contains` to see if a value is not
    /// present on `HashSet` followed by a `insert`.
    ///
    /// ### Why is this bad?
    /// Using just `insert` and checking the returned `bool` is more efficient.
    ///
    /// ### Example
    /// ```rust
    /// use std::collections::HashSet;
    /// let mut set = HashSet::new();
    /// let value = 5;
    /// if !set.contains(&value) {
    ///     set.insert(value);
    ///     println!("inserted {value:?}");
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::collections::HashSet;
    /// let mut set = HashSet::new();
    /// let value = 5;
    /// if set.insert(&value) {
    ///     println!("inserted {value:?}");
    /// }
    /// ```
    #[clippy::version = "1.80.0"]
    pub HASHSET_INSERT_AFTER_CONTAINS,
    nursery,
    "unnecessary call to `HashSet::contains` followed by `HashSet::insert`"
}

declare_lint_pass!(HashsetInsertAfterContains => [HASHSET_INSERT_AFTER_CONTAINS]);

impl<'tcx> LateLintPass<'tcx> for HashsetInsertAfterContains {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion()
            && let Some(higher::If {
                cond: cond_expr,
                then: then_expr,
                ..
            }) = higher::If::hir(expr)
            && let Some(contains_expr) = try_parse_contains(cx, cond_expr)
            && find_insert_calls(cx, &contains_expr, then_expr)
        {
            span_lint_and_then(
                cx,
                HASHSET_INSERT_AFTER_CONTAINS,
                expr.span,
                "usage of `HashSet::insert` after `HashSet::contains`",
                |diag| {
                    diag.note("`HashSet::insert` returns whether it was inserted")
                        .span_help(contains_expr.span, "remove the `HashSet::contains` call");
                },
            );
        }
    }
}

struct ContainsExpr<'tcx> {
    receiver: &'tcx Expr<'tcx>,
    value: &'tcx Expr<'tcx>,
    span: Span,
}
fn try_parse_contains<'tcx>(cx: &LateContext<'_>, expr: &'tcx Expr<'_>) -> Option<ContainsExpr<'tcx>> {
    let expr = peel_hir_expr_while(expr, |e| {
        if let ExprKind::Unary(UnOp::Not, e) = e.kind {
            Some(e)
        } else {
            None
        }
    });
    if let ExprKind::MethodCall(path, receiver, [value], span) = expr.kind {
        let value = value.peel_borrows();
        let receiver = receiver.peel_borrows();
        let receiver_ty = cx.typeck_results().expr_ty(receiver).peel_refs();
        if value.span.eq_ctxt(expr.span)
            && is_type_diagnostic_item(cx, receiver_ty, sym::HashSet)
            && path.ident.name == sym!(contains)
        {
            return Some(ContainsExpr { receiver, value, span });
        }
    }
    None
}

struct InsertExpr<'tcx> {
    receiver: &'tcx Expr<'tcx>,
    value: &'tcx Expr<'tcx>,
}
fn try_parse_insert<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<InsertExpr<'tcx>> {
    if let ExprKind::MethodCall(path, receiver, [value], _) = expr.kind {
        let value = value.peel_borrows();
        let value = peel_hir_expr_while(value, |e| {
            if let ExprKind::Unary(UnOp::Deref, e) = e.kind {
                Some(e)
            } else {
                None
            }
        });

        let receiver_ty = cx.typeck_results().expr_ty(receiver).peel_refs();
        if is_type_diagnostic_item(cx, receiver_ty, sym::HashSet) && path.ident.name == sym!(insert) {
            Some(InsertExpr { receiver, value })
        } else {
            None
        }
    } else {
        None
    }
}

fn find_insert_calls<'tcx>(cx: &LateContext<'tcx>, contains_expr: &ContainsExpr<'tcx>, expr: &'tcx Expr<'_>) -> bool {
    for_each_expr(expr, |e| {
        if let Some(insert_expr) = try_parse_insert(cx, e)
            && SpanlessEq::new(cx).eq_expr(contains_expr.receiver, insert_expr.receiver)
            && SpanlessEq::new(cx).eq_expr(contains_expr.value, insert_expr.value)
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}
