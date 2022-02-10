use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{indent_of, snippet_block, snippet_opt, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, is_refutable, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Arm, Expr, ExprKind, Local, Node, PatKind};
use rustc_lint::LateContext;

use super::MATCH_SINGLE_BINDING;

#[allow(clippy::too_many_lines)]
pub(crate) fn check<'a>(cx: &LateContext<'a>, ex: &Expr<'a>, arms: &[Arm<'_>], expr: &Expr<'_>) {
    if expr.span.from_expansion() || arms.len() != 1 || is_refutable(cx, arms[0].pat) {
        return;
    }

    // HACK:
    // This is a hack to deal with arms that are excluded by macros like `#[cfg]`. It is only used here
    // to prevent false positives as there is currently no better way to detect if code was excluded by
    // a macro. See PR #6435
    if_chain! {
        if let Some(match_snippet) = snippet_opt(cx, expr.span);
        if let Some(arm_snippet) = snippet_opt(cx, arms[0].span);
        if let Some(ex_snippet) = snippet_opt(cx, ex.span);
        let rest_snippet = match_snippet.replace(&arm_snippet, "").replace(&ex_snippet, "");
        if rest_snippet.contains("=>");
        then {
            // The code it self contains another thick arrow "=>"
            // -> Either another arm or a comment
            return;
        }
    }

    let matched_vars = ex.span;
    let bind_names = arms[0].pat.span;
    let match_body = peel_blocks(arms[0].body);
    let mut snippet_body = if match_body.span.from_expansion() {
        Sugg::hir_with_macro_callsite(cx, match_body, "..").to_string()
    } else {
        snippet_block(cx, match_body.span, "..", Some(expr.span)).to_string()
    };

    // Do we need to add ';' to suggestion ?
    match match_body.kind {
        ExprKind::Block(block, _) => {
            // macro + expr_ty(body) == ()
            if block.span.from_expansion() && cx.typeck_results().expr_ty(match_body).is_unit() {
                snippet_body.push(';');
            }
        },
        _ => {
            // expr_ty(body) == ()
            if cx.typeck_results().expr_ty(match_body).is_unit() {
                snippet_body.push(';');
            }
        },
    }

    let mut applicability = Applicability::MaybeIncorrect;
    match arms[0].pat.kind {
        PatKind::Binding(..) | PatKind::Tuple(_, _) | PatKind::Struct(..) => {
            // If this match is in a local (`let`) stmt
            let (target_span, sugg) = if let Some(parent_let_node) = opt_parent_let(cx, ex) {
                (
                    parent_let_node.span,
                    format!(
                        "let {} = {};\n{}let {} = {};",
                        snippet_with_applicability(cx, bind_names, "..", &mut applicability),
                        snippet_with_applicability(cx, matched_vars, "..", &mut applicability),
                        " ".repeat(indent_of(cx, expr.span).unwrap_or(0)),
                        snippet_with_applicability(cx, parent_let_node.pat.span, "..", &mut applicability),
                        snippet_body
                    ),
                )
            } else {
                // If we are in closure, we need curly braces around suggestion
                let mut indent = " ".repeat(indent_of(cx, ex.span).unwrap_or(0));
                let (mut cbrace_start, mut cbrace_end) = ("".to_string(), "".to_string());
                if let Some(parent_expr) = get_parent_expr(cx, expr) {
                    if let ExprKind::Closure(..) = parent_expr.kind {
                        cbrace_end = format!("\n{}}}", indent);
                        // Fix body indent due to the closure
                        indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
                        cbrace_start = format!("{{\n{}", indent);
                    }
                }
                // If the parent is already an arm, and the body is another match statement,
                // we need curly braces around suggestion
                let parent_node_id = cx.tcx.hir().get_parent_node(expr.hir_id);
                if let Node::Arm(arm) = &cx.tcx.hir().get(parent_node_id) {
                    if let ExprKind::Match(..) = arm.body.kind {
                        cbrace_end = format!("\n{}}}", indent);
                        // Fix body indent due to the match
                        indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
                        cbrace_start = format!("{{\n{}", indent);
                    }
                }
                (
                    expr.span,
                    format!(
                        "{}let {} = {};\n{}{}{}",
                        cbrace_start,
                        snippet_with_applicability(cx, bind_names, "..", &mut applicability),
                        snippet_with_applicability(cx, matched_vars, "..", &mut applicability),
                        indent,
                        snippet_body,
                        cbrace_end
                    ),
                )
            };
            span_lint_and_sugg(
                cx,
                MATCH_SINGLE_BINDING,
                target_span,
                "this match could be written as a `let` statement",
                "consider using `let` statement",
                sugg,
                applicability,
            );
        },
        PatKind::Wild => {
            if ex.can_have_side_effects() {
                let indent = " ".repeat(indent_of(cx, expr.span).unwrap_or(0));
                let sugg = format!(
                    "{};\n{}{}",
                    snippet_with_applicability(cx, ex.span, "..", &mut applicability),
                    indent,
                    snippet_body
                );
                span_lint_and_sugg(
                    cx,
                    MATCH_SINGLE_BINDING,
                    expr.span,
                    "this match could be replaced by its scrutinee and body",
                    "consider using the scrutinee and body instead",
                    sugg,
                    applicability,
                );
            } else {
                span_lint_and_sugg(
                    cx,
                    MATCH_SINGLE_BINDING,
                    expr.span,
                    "this match could be replaced by its body itself",
                    "consider using the match body instead",
                    snippet_body,
                    Applicability::MachineApplicable,
                );
            }
        },
        _ => (),
    }
}

/// Returns true if the `ex` match expression is in a local (`let`) statement
fn opt_parent_let<'a>(cx: &LateContext<'a>, ex: &Expr<'a>) -> Option<&'a Local<'a>> {
    let map = &cx.tcx.hir();
    if_chain! {
        if let Some(Node::Expr(parent_arm_expr)) = map.find(map.get_parent_node(ex.hir_id));
        if let Some(Node::Local(parent_let_expr)) = map.find(map.get_parent_node(parent_arm_expr.hir_id));
        then {
            return Some(parent_let_expr);
        }
    }
    None
}
