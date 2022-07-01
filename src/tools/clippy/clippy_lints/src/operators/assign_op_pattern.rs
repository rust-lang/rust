use clippy_utils::binop_traits;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::implements_trait;
use clippy_utils::{eq_expr_value, trait_ref_of_method};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_lint::LateContext;

use super::ASSIGN_OP_PATTERN;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    assignee: &'tcx hir::Expr<'_>,
    e: &'tcx hir::Expr<'_>,
) {
    if let hir::ExprKind::Binary(op, l, r) = &e.kind {
        let lint = |assignee: &hir::Expr<'_>, rhs: &hir::Expr<'_>| {
            let ty = cx.typeck_results().expr_ty(assignee);
            let rty = cx.typeck_results().expr_ty(rhs);
            if_chain! {
                if let Some((_, lang_item)) = binop_traits(op.node);
                if let Ok(trait_id) = cx.tcx.lang_items().require(lang_item);
                let parent_fn = cx.tcx.hir().get_parent_item(e.hir_id);
                if trait_ref_of_method(cx, parent_fn)
                    .map_or(true, |t| t.path.res.def_id() != trait_id);
                if implements_trait(cx, ty, trait_id, &[rty.into()]);
                then {
                    span_lint_and_then(
                        cx,
                        ASSIGN_OP_PATTERN,
                        expr.span,
                        "manual implementation of an assign operation",
                        |diag| {
                            if let (Some(snip_a), Some(snip_r)) =
                                (snippet_opt(cx, assignee.span), snippet_opt(cx, rhs.span))
                            {
                                diag.span_suggestion(
                                    expr.span,
                                    "replace it with",
                                    format!("{} {}= {}", snip_a, op.node.as_str(), snip_r),
                                    Applicability::MachineApplicable,
                                );
                            }
                        },
                    );
                }
            }
        };

        let mut visitor = ExprVisitor {
            assignee,
            counter: 0,
            cx,
        };

        walk_expr(&mut visitor, e);

        if visitor.counter == 1 {
            // a = a op b
            if eq_expr_value(cx, assignee, l) {
                lint(assignee, r);
            }
            // a = b commutative_op a
            // Limited to primitive type as these ops are know to be commutative
            if eq_expr_value(cx, assignee, r) && cx.typeck_results().expr_ty(assignee).is_primitive_ty() {
                match op.node {
                    hir::BinOpKind::Add
                    | hir::BinOpKind::Mul
                    | hir::BinOpKind::And
                    | hir::BinOpKind::Or
                    | hir::BinOpKind::BitXor
                    | hir::BinOpKind::BitAnd
                    | hir::BinOpKind::BitOr => {
                        lint(assignee, l);
                    },
                    _ => {},
                }
            }
        }
    }
}

struct ExprVisitor<'a, 'tcx> {
    assignee: &'a hir::Expr<'a>,
    counter: u8,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for ExprVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if eq_expr_value(self.cx, self.assignee, expr) {
            self.counter += 1;
        }

        walk_expr(self, expr);
    }
}
