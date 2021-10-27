use super::SINGLE_ELEMENT_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::single_segment_path;
use clippy_utils::source::{indent_of, snippet};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Pat, PatKind};
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
        ExprKind::MethodCall(method, _, args, _) if args.len() == 1 && method.ident.name == rustc_span::sym::iter => {
            &args[0]
        },
        _ => return,
    };
    if_chain! {
        if let PatKind::Binding(.., target, _) = pat.kind;
        if let ExprKind::Array([arg_expression]) = arg_expr.kind;
        if let ExprKind::Path(ref list_item) = arg_expression.kind;
        if let Some(list_item_name) = single_segment_path(list_item).map(|ps| ps.ident.name);
        if let ExprKind::Block(block, _) = body.kind;
        if !block.stmts.is_empty();

        then {
            let mut block_str = snippet(cx, block.span, "..").into_owned();
            block_str.remove(0);
            block_str.pop();


            span_lint_and_sugg(
                cx,
                SINGLE_ELEMENT_LOOP,
                expr.span,
                "for loop over a single element",
                "try",
                format!("{{\n{}let {} = &{};{}}}", " ".repeat(indent_of(cx, block.stmts[0].span).unwrap_or(0)), target.name, list_item_name, block_str),
                Applicability::MachineApplicable
            )
        }
    }
}
