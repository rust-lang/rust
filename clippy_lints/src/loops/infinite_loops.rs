use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_lint_allowed;
use hir::intravisit::{walk_expr, Visitor};
use hir::{Block, Destination, Expr, ExprKind, FnRetTy, Ty, TyKind};
use rustc_ast::Label;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;

use super::INFINITE_LOOPS;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    loop_block: &Block<'_>,
    label: Option<Label>,
    parent_fn_ret_ty: FnRetTy<'_>,
) {
    if is_lint_allowed(cx, INFINITE_LOOPS, expr.hir_id)
        || matches!(
            parent_fn_ret_ty,
            FnRetTy::Return(Ty {
                kind: TyKind::Never,
                ..
            })
        )
    {
        return;
    }

    // First, find any `break` or `return` without entering any inner loop,
    // then, find `return` or labeled `break` which breaks this loop with entering inner loop,
    // otherwise this loop is a infinite loop.
    let mut direct_br_or_ret_finder = BreakOrRetFinder::default();
    direct_br_or_ret_finder.visit_block(loop_block);

    let is_finite_loop = direct_br_or_ret_finder.found || {
        let mut inner_br_or_ret_finder = BreakOrRetFinder {
            label,
            enter_nested_loop: true,
            ..Default::default()
        };
        inner_br_or_ret_finder.visit_block(loop_block);
        inner_br_or_ret_finder.found
    };

    if !is_finite_loop {
        span_lint_and_then(cx, INFINITE_LOOPS, expr.span, "infinite loop detected", |diag| {
            if let FnRetTy::DefaultReturn(ret_span) = parent_fn_ret_ty {
                diag.span_suggestion(
                    ret_span,
                    "if this is intentional, consider specifing `!` as function return",
                    " -> !",
                    Applicability::MaybeIncorrect,
                );
            } else {
                diag.span_help(
                    expr.span,
                    "if this is not intended, add a `break` or `return` condition in this loop",
                );
            }
        });
    }
}

#[derive(Default)]
struct BreakOrRetFinder {
    label: Option<Label>,
    found: bool,
    enter_nested_loop: bool,
}

impl<'hir> Visitor<'hir> for BreakOrRetFinder {
    fn visit_expr(&mut self, ex: &'hir Expr<'_>) {
        match &ex.kind {
            ExprKind::Break(Destination { label, .. }, ..) => {
                // When entering nested loop, only by breaking this loop's label
                // would be considered as exiting this loop.
                if self.enter_nested_loop {
                    if label.is_some() && *label == self.label {
                        self.found = true;
                    }
                } else {
                    self.found = true;
                }
            },
            ExprKind::Ret(..) => self.found = true,
            ExprKind::Loop(..) if !self.enter_nested_loop => (),
            _ => walk_expr(self, ex),
        }
    }
}
