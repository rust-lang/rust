use super::SINGLE_ELEMENT_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, snippet, snippet_with_applicability};
use clippy_utils::visitors::contains_break_or_continue;
use rustc_ast::Mutability;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Pat, PatKind, is_range_literal};
use rustc_lint::LateContext;
use rustc_span::edition::Edition;
use rustc_span::sym;

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
        ) if method.ident.name == sym::iter => (arg, "&"),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
            [],
            _,
        ) if method.ident.name == sym::iter_mut => (arg, "&mut "),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([arg]),
                ..
            },
            [],
            _,
        ) if method.ident.name == sym::into_iter => (arg, ""),
        // Only check for arrays edition 2021 or later, as this case will trigger a compiler error otherwise.
        ExprKind::Array([arg]) if cx.tcx.sess.edition() >= Edition::Edition2021 => (arg, ""),
        _ => return,
    };
    if let ExprKind::Block(block, _) = body.kind
        && !block.stmts.is_empty()
        && !contains_break_or_continue(body)
    {
        let mut applicability = Applicability::MachineApplicable;
        let mut pat_snip = snippet_with_applicability(cx, pat.span, "..", &mut applicability);
        if matches!(pat.kind, PatKind::Or(..)) {
            pat_snip = format!("({pat_snip})").into();
        }
        let mut arg_snip = snippet_with_applicability(cx, arg_expression.span, "..", &mut applicability);
        let mut block_str = snippet_with_applicability(cx, block.span, "..", &mut applicability).into_owned();
        block_str.remove(0);
        block_str.pop();
        let indent = " ".repeat(indent_of(cx, block.stmts[0].span).unwrap_or(0));

        // Reference iterator from `&(mut) []` or `[].iter(_mut)()`.
        if !prefix.is_empty()
            && (
                // Precedence of internal expression is less than or equal to precedence of `&expr`.
                cx.precedence(arg_expression) <= ExprPrecedence::Prefix || is_range_literal(arg_expression)
            )
        {
            arg_snip = format!("({arg_snip})").into();
        }

        if clippy_utils::higher::Range::hir(arg_expression).is_some() {
            let range_expr = snippet(cx, arg_expression.span, "?").to_string();

            let sugg = snippet(cx, arg_expression.span, "..");
            span_lint_and_sugg(
                cx,
                SINGLE_ELEMENT_LOOP,
                arg.span,
                format!("this loops only once with `{pat_snip}` being `{range_expr}`"),
                "did you mean to iterate over the range instead?",
                sugg.to_string(),
                Applicability::Unspecified,
            );
        } else {
            span_lint_and_sugg(
                cx,
                SINGLE_ELEMENT_LOOP,
                expr.span,
                "for loop over a single element",
                "try",
                format!("{{\n{indent}let {pat_snip} = {prefix}{arg_snip};{block_str}}}"),
                applicability,
            );
        }
    }
}
