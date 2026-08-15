use super::SINGLE_ELEMENT_LOOP;
use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
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
    let Some((arg_expression, prefix)) = extract_single_element(cx, arg) else {
        return;
    };

    if let ExprKind::Block(block, _) = body.kind
        && (!block.stmts.is_empty() || block.expr.is_some())
        && !contains_break_or_continue(body)
    {
        emit_lint(cx, pat, arg, expr, arg_expression, prefix, block);
    }
}

fn extract_single_element<'tcx>(
    cx: &LateContext<'tcx>,
    arg: &'tcx Expr<'_>,
) -> Option<(&'tcx Expr<'tcx>, &'static str)> {
    match arg.kind {
        ExprKind::AddrOf(
            BorrowKind::Ref,
            Mutability::Not,
            Expr {
                kind: ExprKind::Array([inner]),
                ..
            },
        ) => Some((inner, "&")),
        ExprKind::AddrOf(
            BorrowKind::Ref,
            Mutability::Mut,
            Expr {
                kind: ExprKind::Array([inner]),
                ..
            },
        ) => Some((inner, "&mut ")),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([inner]),
                ..
            },
            [],
            _,
        ) if method.ident.name == sym::iter => Some((inner, "&")),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([inner]),
                ..
            },
            [],
            _,
        ) if method.ident.name == sym::iter_mut => Some((inner, "&mut ")),
        ExprKind::MethodCall(
            method,
            Expr {
                kind: ExprKind::Array([inner]),
                ..
            },
            [],
            _,
        ) if method.ident.name == sym::into_iter => Some((inner, "")),
        // Only check for arrays edition 2021 or later, as this case will trigger a compiler error otherwise.
        ExprKind::Array([inner]) if cx.tcx.sess.edition() >= Edition::Edition2021 => Some((inner, "")),
        _ => None,
    }
}

fn emit_lint<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    arg: &'tcx Expr<'_>,
    expr: &'tcx Expr<'_>,
    arg_expression: &'tcx Expr<'_>,
    prefix: &str,
    block: &rustc_hir::Block<'_>,
) {
    let mut applicability = Applicability::MachineApplicable;
    let mut pat_snip = snippet_with_applicability(cx, pat.span, "..", &mut applicability);
    if matches!(pat.kind, PatKind::Or(..)) {
        pat_snip = format!("({pat_snip})").into();
    }
    let mut arg_snip = snippet_with_applicability(cx, arg_expression.span, "..", &mut applicability);
    let mut block_str = snippet_with_applicability(cx, block.span, "..", &mut applicability).into_owned();
    block_str.remove(0);
    block_str.pop();
    let indent = if let Some(first_stmt) = block.stmts.first() {
        " ".repeat(indent_of(cx, first_stmt.span).unwrap_or(0))
    } else if let Some(block_expr) = block.expr {
        " ".repeat(indent_of(cx, block_expr.span).unwrap_or(0))
    } else {
        String::new()
    };

    // Reference iterator from `&(mut) []` or `[].iter(_mut)()`.
    if !prefix.is_empty()
        && (
            // Precedence of internal expression is less than or equal to precedence of `&expr`.
            cx.precedence(arg_expression) <= ExprPrecedence::Prefix || is_range_literal(arg_expression)
        )
    {
        arg_snip = format!("({arg_snip})").into();
    }

    if let Some(range) = clippy_utils::higher::Range::hir(cx, arg_expression) {
        if range.start.is_some() {
            // Only suggest iterating over ranges that have a start value.
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
            span_lint(cx, SINGLE_ELEMENT_LOOP, expr.span, "for loop over a single element");
        }
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
