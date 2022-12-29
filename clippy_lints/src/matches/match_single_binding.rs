use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::HirNode;
use clippy_utils::source::{indent_of, snippet, snippet_block, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::{get_parent_expr, is_refutable, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::{Arm, Expr, ExprKind, Node, PatKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::MATCH_SINGLE_BINDING;

enum AssignmentExpr {
    Assign { span: Span, match_span: Span },
    Local { span: Span, pat_span: Span },
}

#[expect(clippy::too_many_lines)]
pub(crate) fn check<'a>(cx: &LateContext<'a>, ex: &Expr<'a>, arms: &[Arm<'_>], expr: &Expr<'a>) {
    if expr.span.from_expansion() || arms.len() != 1 || is_refutable(cx, arms[0].pat) {
        return;
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
    if let ExprKind::Block(block, _) = match_body.kind {
        // macro + expr_ty(body) == ()
        if block.span.from_expansion() && cx.typeck_results().expr_ty(match_body).is_unit() {
            snippet_body.push(';');
        }
    }

    let mut applicability = Applicability::MaybeIncorrect;
    match arms[0].pat.kind {
        PatKind::Binding(..) | PatKind::Tuple(_, _) | PatKind::Struct(..) => {
            let (target_span, sugg) = match opt_parent_assign_span(cx, ex) {
                Some(AssignmentExpr::Assign { span, match_span }) => {
                    let sugg = sugg_with_curlies(
                        cx,
                        (ex, expr),
                        (bind_names, matched_vars),
                        &snippet_body,
                        &mut applicability,
                        Some(span),
                        true,
                    );

                    span_lint_and_sugg(
                        cx,
                        MATCH_SINGLE_BINDING,
                        span.to(match_span),
                        "this assignment could be simplified",
                        "consider removing the `match` expression",
                        sugg,
                        applicability,
                    );

                    return;
                },
                Some(AssignmentExpr::Local { span, pat_span }) => (
                    span,
                    format!(
                        "let {} = {};\n{}let {} = {snippet_body};",
                        snippet_with_applicability(cx, bind_names, "..", &mut applicability),
                        snippet_with_applicability(cx, matched_vars, "..", &mut applicability),
                        " ".repeat(indent_of(cx, expr.span).unwrap_or(0)),
                        snippet_with_applicability(cx, pat_span, "..", &mut applicability)
                    ),
                ),
                None => {
                    let sugg = sugg_with_curlies(
                        cx,
                        (ex, expr),
                        (bind_names, matched_vars),
                        &snippet_body,
                        &mut applicability,
                        None,
                        true,
                    );
                    (expr.span, sugg)
                },
            };

            span_lint_and_sugg(
                cx,
                MATCH_SINGLE_BINDING,
                target_span,
                "this match could be written as a `let` statement",
                "consider using a `let` statement",
                sugg,
                applicability,
            );
        },
        PatKind::Wild => {
            if ex.can_have_side_effects() {
                let sugg = sugg_with_curlies(
                    cx,
                    (ex, expr),
                    (bind_names, matched_vars),
                    &snippet_body,
                    &mut applicability,
                    None,
                    false,
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

/// Returns true if the `ex` match expression is in a local (`let`) or assign expression
fn opt_parent_assign_span<'a>(cx: &LateContext<'a>, ex: &Expr<'a>) -> Option<AssignmentExpr> {
    let map = &cx.tcx.hir();

    if let Some(Node::Expr(parent_arm_expr)) = map.find(map.get_parent_node(ex.hir_id)) {
        return match map.find(map.get_parent_node(parent_arm_expr.hir_id)) {
            Some(Node::Local(parent_let_expr)) => Some(AssignmentExpr::Local {
                span: parent_let_expr.span,
                pat_span: parent_let_expr.pat.span(),
            }),
            Some(Node::Expr(Expr {
                kind: ExprKind::Assign(parent_assign_expr, match_expr, _),
                ..
            })) => Some(AssignmentExpr::Assign {
                span: parent_assign_expr.span,
                match_span: match_expr.span,
            }),
            _ => None,
        };
    }

    None
}

fn sugg_with_curlies<'a>(
    cx: &LateContext<'a>,
    (ex, match_expr): (&Expr<'a>, &Expr<'a>),
    (bind_names, matched_vars): (Span, Span),
    snippet_body: &str,
    applicability: &mut Applicability,
    assignment: Option<Span>,
    needs_var_binding: bool,
) -> String {
    let mut indent = " ".repeat(indent_of(cx, ex.span).unwrap_or(0));

    let (mut cbrace_start, mut cbrace_end) = (String::new(), String::new());
    if let Some(parent_expr) = get_parent_expr(cx, match_expr) {
        if let ExprKind::Closure { .. } = parent_expr.kind {
            cbrace_end = format!("\n{indent}}}");
            // Fix body indent due to the closure
            indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
            cbrace_start = format!("{{\n{indent}");
        }
    }

    // If the parent is already an arm, and the body is another match statement,
    // we need curly braces around suggestion
    let parent_node_id = cx.tcx.hir().get_parent_node(match_expr.hir_id);
    if let Node::Arm(arm) = &cx.tcx.hir().get(parent_node_id) {
        if let ExprKind::Match(..) = arm.body.kind {
            cbrace_end = format!("\n{indent}}}");
            // Fix body indent due to the match
            indent = " ".repeat(indent_of(cx, bind_names).unwrap_or(0));
            cbrace_start = format!("{{\n{indent}");
        }
    }

    let assignment_str = assignment.map_or_else(String::new, |span| {
        let mut s = snippet(cx, span, "..").to_string();
        s.push_str(" = ");
        s
    });

    let scrutinee = if needs_var_binding {
        format!(
            "let {} = {}",
            snippet_with_applicability(cx, bind_names, "..", applicability),
            snippet_with_applicability(cx, matched_vars, "..", applicability)
        )
    } else {
        snippet_with_applicability(cx, matched_vars, "..", applicability).to_string()
    };

    format!("{cbrace_start}{scrutinee};\n{indent}{assignment_str}{snippet_body}{cbrace_end}")
}
