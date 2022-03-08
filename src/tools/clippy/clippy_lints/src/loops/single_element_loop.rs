use super::SINGLE_ELEMENT_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, snippet_with_applicability};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Pat};
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
) {
    let arg_expr = match arg.kind {
        ExprKind::AddrOf(BorrowKind::Ref, _, ref_arg) => ref_arg,
        ExprKind::MethodCall(method, [arg], _) if method.ident.name == rustc_span::sym::iter => arg,
        _ => return,
    };
    if_chain! {
        if let ExprKind::Array([arg_expression]) = arg_expr.kind;
        if let ExprKind::Block(block, _) = body.kind;
        if !block.stmts.is_empty();
        then {
            let mut applicability = Applicability::MachineApplicable;
            let pat_snip = snippet_with_applicability(cx, pat.span, "..", &mut applicability);
            let arg_snip = snippet_with_applicability(cx, arg_expression.span, "..", &mut applicability);
            let mut block_str = snippet_with_applicability(cx, block.span, "..", &mut applicability).into_owned();
            block_str.remove(0);
            block_str.pop();
            let indent = " ".repeat(indent_of(cx, block.stmts[0].span).unwrap_or(0));

            span_lint_and_sugg(
                cx,
                SINGLE_ELEMENT_LOOP,
                expr.span,
                "for loop over a single element",
                "try",
                format!("{{\n{}let {} = &{};{}}}", indent, pat_snip, arg_snip, block_str),
                applicability,
            )
        }
    }
}
