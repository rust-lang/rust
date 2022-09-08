use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_trait_method, path_to_local_id, peel_blocks, strip_pat_refs};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::PatKind;
use rustc_lint::LateContext;
use rustc_span::{source_map::Span, sym};

use super::UNNECESSARY_FOLD;

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    init: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
) {
    fn check_fold_with_op(
        cx: &LateContext<'_>,
        expr: &hir::Expr<'_>,
        acc: &hir::Expr<'_>,
        fold_span: Span,
        op: hir::BinOpKind,
        replacement_method_name: &str,
        replacement_has_args: bool,
    ) {
        if_chain! {
            // Extract the body of the closure passed to fold
            if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = acc.kind;
            let closure_body = cx.tcx.hir().body(body);
            let closure_expr = peel_blocks(closure_body.value);

            // Check if the closure body is of the form `acc <op> some_expr(x)`
            if let hir::ExprKind::Binary(ref bin_op, left_expr, right_expr) = closure_expr.kind;
            if bin_op.node == op;

            // Extract the names of the two arguments to the closure
            if let [param_a, param_b] = closure_body.params;
            if let PatKind::Binding(_, first_arg_id, ..) = strip_pat_refs(param_a.pat).kind;
            if let PatKind::Binding(_, second_arg_id, second_arg_ident, _) = strip_pat_refs(param_b.pat).kind;

            if path_to_local_id(left_expr, first_arg_id);
            if replacement_has_args || path_to_local_id(right_expr, second_arg_id);

            then {
                let mut applicability = Applicability::MachineApplicable;
                let sugg = if replacement_has_args {
                    format!(
                        "{replacement}(|{s}| {r})",
                        replacement = replacement_method_name,
                        s = second_arg_ident,
                        r = snippet_with_applicability(cx, right_expr.span, "EXPR", &mut applicability),
                    )
                } else {
                    format!(
                        "{replacement}()",
                        replacement = replacement_method_name,
                    )
                };

                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_FOLD,
                    fold_span.with_hi(expr.span.hi()),
                    // TODO #2371 don't suggest e.g., .any(|x| f(x)) if we can suggest .any(f)
                    "this `.fold` can be written more succinctly using another method",
                    "try",
                    sugg,
                    applicability,
                );
            }
        }
    }

    // Check that this is a call to Iterator::fold rather than just some function called fold
    if !is_trait_method(cx, expr, sym::Iterator) {
        return;
    }

    // Check if the first argument to .fold is a suitable literal
    if let hir::ExprKind::Lit(ref lit) = init.kind {
        match lit.node {
            ast::LitKind::Bool(false) => check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Or, "any", true),
            ast::LitKind::Bool(true) => check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::And, "all", true),
            ast::LitKind::Int(0, _) => check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Add, "sum", false),
            ast::LitKind::Int(1, _) => {
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Mul, "product", false);
            },
            _ => (),
        }
    }
}
