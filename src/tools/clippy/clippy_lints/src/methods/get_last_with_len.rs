use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::SpanlessEq;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::source_map::Spanned;
use rustc_span::sym;

use super::GET_LAST_WITH_LEN;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, arg: &Expr<'_>) {
    // Argument to "get" is a subtraction
    if let ExprKind::Binary(
        Spanned {
            node: BinOpKind::Sub, ..
        },
        lhs,
        rhs,
    ) = arg.kind

        // LHS of subtraction is "x.len()"
        && let ExprKind::MethodCall(lhs_path, lhs_recv, [], _) = &lhs.kind
        && lhs_path.ident.name == sym::len

        // RHS of subtraction is 1
        && let ExprKind::Lit(rhs_lit) = &rhs.kind
        && let LitKind::Int(1, ..) = rhs_lit.node

        // check that recv == lhs_recv `recv.get(lhs_recv.len() - 1)`
        && SpanlessEq::new(cx).eq_expr(recv, lhs_recv)
        && !recv.can_have_side_effects()
    {
        let method = match cx.typeck_results().expr_ty_adjusted(recv).peel_refs().kind() {
            ty::Adt(def, _) if cx.tcx.is_diagnostic_item(sym::VecDeque, def.did()) => "back",
            ty::Slice(_) => "last",
            _ => return,
        };

        let mut applicability = Applicability::MachineApplicable;
        let recv_snippet = snippet_with_applicability(cx, recv.span, "_", &mut applicability);

        span_lint_and_sugg(
            cx,
            GET_LAST_WITH_LEN,
            expr.span,
            &format!("accessing last element with `{recv_snippet}.get({recv_snippet}.len() - 1)`"),
            "try",
            format!("{recv_snippet}.{method}()"),
            applicability,
        );
    }
}
