use super::SINGLE_ELEMENT_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, snippet_with_applicability};
use clippy_utils::visitors::for_each_expr;
use if_chain::if_chain;
use rustc_ast::util::parser::PREC_PREFIX;
use rustc_ast::Mutability;
use rustc_errors::Applicability;
use rustc_hir::{is_range_literal, BorrowKind, Expr, ExprKind, Pat};
use rustc_lint::LateContext;
use rustc_span::edition::Edition;
use std::ops::ControlFlow;

fn contains_break_or_continue(expr: &Expr<'_>) -> bool {
    for_each_expr(expr, |e| {
        if matches!(e.kind, ExprKind::Break(..) | ExprKind::Continue(..)) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    body: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
) {
    let (arg_expression, prefix) = match arg.kind {
        ExprKind::AddrOf(
            BorrowKind::Ref,
            Mutability::Not,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
        ) => (arg, "&"),
        ExprKind::AddrOf(
            BorrowKind::Ref,
            Mutability::Mut,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
        ) => (arg, "&mut "),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
            [],
            _,
        ) if method.ident.name == rustc_span::sym::iter => (arg, "&"),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
            [],
            _,
        ) if method.ident.name.as_str() == "iter_mut" => (arg, "&mut "),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
            [],
            _,
        ) if method.ident.name == rustc_span::sym::into_iter => (arg, ""),
        // Only check for arrays edition 2021 or later, as this case will trigger a compiler error otherwise.
        ExprKind::Array([arg]) if cx.tcx.sess.edition() >= Edition::Edition2021 => (arg, ""),
        _ => return,
    };
    if_chain! {
        if let ExprKind::Block(block, _) = body.kind;
        if !block.stmts.is_empty();
        if !contains_break_or_continue(body);
        then {
            let mut applicability = Applicability::MachineApplicable;
            let pat_snip = snippet_with_applicability(cx, pat.span, "..", &mut applicability);
            let mut arg_snip = snippet_with_applicability(cx, arg_expression.span, "..", &mut applicability);
            let mut block_str = snippet_with_applicability(cx, block.span, "..", &mut applicability).into_owned();
            block_str.remove(0);
            block_str.pop();
            let indent = " ".repeat(indent_of(cx, block.stmts[0].span).unwrap_or(0));

            // Reference iterator from `&(mut) []` or `[].iter(_mut)()`.
            if !prefix.is_empty() && (
                // Precedence of internal expression is less than or equal to precedence of `&expr`.
                arg_expression.precedence().order() <= PREC_PREFIX || is_range_literal(arg_expression)
            ) {
                arg_snip = format!("({arg_snip})").into();
            }

            span_lint_and_sugg(
                cx,
                SINGLE_ELEMENT_LOOP,
                expr.span,
                "for loop over a single element",
                "try",
                format!("{{\n{indent}let {pat_snip} = {prefix}{arg_snip};{block_str}}}"),
                applicability,
            )
        }
    }
}
