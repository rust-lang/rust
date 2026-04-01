use std::iter;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, indent_of, reindent_multiline};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::expr_type_is_certain;
use clippy_utils::{is_expr_default, is_from_proc_macro};
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, MatchSource, Node, StmtKind};
use rustc_lint::LateContext;
use rustc_span::SyntaxContext;

use super::{UNIT_ARG, utils};

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
    if expr.span.from_expansion() {
        return;
    }

    // apparently stuff in the desugaring of `?` can trigger this
    // so check for that here
    // only the calls to `Try::from_error` is marked as desugared,
    // so we need to check both the current Expr and its parent.
    if is_questionmark_desugar_marked_call(expr) {
        return;
    }
    if let Node::Expr(parent_expr) = cx.tcx.parent_hir_node(expr.hir_id)
        && is_questionmark_desugar_marked_call(parent_expr)
    {
        return;
    }

    let (receiver, args) = match expr.kind {
        ExprKind::Call(_, args) => (None, args),
        ExprKind::MethodCall(_, receiver, args, _) => (Some(receiver), args),
        _ => return,
    };

    let args_to_recover = receiver
        .into_iter()
        .chain(args)
        .filter(|arg| {
            if cx.typeck_results().expr_ty(arg).is_unit() && !utils::is_unit_literal(arg) {
                !matches!(
                    &arg.kind,
                    ExprKind::Match(.., MatchSource::TryDesugar(_)) | ExprKind::Path(..)
                )
            } else {
                false
            }
        })
        .collect::<Vec<_>>();
    if !args_to_recover.is_empty() && !is_from_proc_macro(cx, expr) {
        lint_unit_args(cx, expr, args_to_recover.as_slice());
    }
}

fn is_questionmark_desugar_marked_call(expr: &Expr<'_>) -> bool {
    use rustc_span::hygiene::DesugaringKind;
    if let ExprKind::Call(callee, _) = expr.kind {
        callee.span.is_desugaring(DesugaringKind::QuestionMark)
    } else {
        false
    }
}

fn lint_unit_args<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, args_to_recover: &[&'tcx Expr<'tcx>]) {
    let mut applicability = Applicability::MachineApplicable;
    let (singular, plural) = if args_to_recover.len() > 1 {
        ("", "s")
    } else {
        ("a ", "")
    };
    span_lint_and_then(
        cx,
        UNIT_ARG,
        expr.span,
        format!("passing {singular}unit value{plural} to a function"),
        |db| {
            let mut or = "";
            args_to_recover
                .iter()
                .filter_map(|arg| {
                    if let ExprKind::Block(block, _) = arg.kind
                        && block.expr.is_none()
                        && let Some(last_stmt) = block.stmts.iter().last()
                        && let StmtKind::Semi(last_expr) = last_stmt.kind
                        && let Some(snip) = last_expr.span.get_source_text(cx)
                    {
                        Some((last_stmt.span, snip))
                    } else {
                        None
                    }
                })
                .for_each(|(span, sugg)| {
                    db.span_suggestion(
                        span,
                        "remove the semicolon from the last statement in the block",
                        sugg.as_str(),
                        Applicability::MaybeIncorrect,
                    );
                    or = "or ";
                    applicability = Applicability::MaybeIncorrect;
                });

            let arg_snippets: Vec<_> = args_to_recover
                .iter()
                // If the argument is from an expansion and is a `Default::default()`, we skip it
                .filter(|arg| !arg.span.from_expansion() || !is_expr_default_nested(cx, arg))
                .filter_map(|arg| get_expr_snippet(cx, arg))
                .collect();

            // If the argument is an empty block or `Default::default()`, we can replace it with `()`.
            let arg_snippets_without_redundant_exprs: Vec<_> = args_to_recover
                .iter()
                .filter(|arg| !is_expr_default_nested(cx, arg) && (arg.span.from_expansion() || !is_empty_block(arg)))
                .filter_map(|arg| get_expr_snippet_with_type_certainty(cx, arg))
                .collect();

            if let Some(call_snippet) = expr.span.get_source_text(cx) {
                if arg_snippets_without_redundant_exprs.is_empty()
                    && let suggestions = args_to_recover
                        .iter()
                        .filter(|arg| !arg.span.from_expansion() || !is_expr_default_nested(cx, arg))
                        .map(|arg| (arg.span.parent_callsite().unwrap_or(arg.span), "()".to_string()))
                        .collect::<Vec<_>>()
                    && !suggestions.is_empty()
                {
                    db.multipart_suggestion(
                        format!("use {singular}unit literal{plural} instead"),
                        suggestions,
                        applicability,
                    );
                } else {
                    let plural = arg_snippets_without_redundant_exprs.len() > 1;
                    let sugg = fmt_stmts_and_call(
                        cx,
                        expr,
                        &call_snippet,
                        arg_snippets,
                        arg_snippets_without_redundant_exprs,
                    );
                    let empty_or_s = if plural { "s" } else { "" };
                    let it_or_them = if plural { "them" } else { "it" };
                    db.span_suggestion(
                        expr.span,
                        format!(
                            "{or}move the expression{empty_or_s} in front of the call and replace {it_or_them} with the unit literal `()`"
                        ),
                        sugg,
                        applicability,
                    );
                }
            }
        },
    );
}

fn is_expr_default_nested<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    is_expr_default(cx, expr)
        || matches!(expr.kind, ExprKind::Block(block, _)
        if block.expr.is_some() && is_expr_default_nested(cx, block.expr.unwrap()))
}

enum MaybeTypeUncertain<'tcx> {
    Certain(Sugg<'tcx>),
    Uncertain(Sugg<'tcx>),
}

impl From<MaybeTypeUncertain<'_>> for String {
    fn from(value: MaybeTypeUncertain<'_>) -> Self {
        match value {
            MaybeTypeUncertain::Certain(sugg) => sugg.to_string(),
            MaybeTypeUncertain::Uncertain(sugg) => format!("let _: () = {sugg}"),
        }
    }
}

fn get_expr_snippet<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<Sugg<'tcx>> {
    let mut app = Applicability::MachineApplicable;
    let snip = Sugg::hir_with_context(cx, expr, SyntaxContext::root(), "..", &mut app);
    if app != Applicability::MachineApplicable {
        return None;
    }

    Some(snip)
}

fn get_expr_snippet_with_type_certainty<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
) -> Option<MaybeTypeUncertain<'tcx>> {
    get_expr_snippet(cx, expr).map(|snip| {
        // If the type of the expression is certain, we can use it directly.
        // Otherwise, we wrap it in a `let _: () = ...` to ensure the type is correct.
        if !expr_type_is_certain(cx, expr) && !is_block_with_no_expr(expr) {
            MaybeTypeUncertain::Uncertain(snip)
        } else {
            MaybeTypeUncertain::Certain(snip)
        }
    })
}

fn is_block_with_no_expr(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Block(Block { expr: None, .. }, _))
}

fn is_empty_block(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Block(
            Block {
                stmts: [],
                expr: None,
                ..
            },
            _,
        )
    )
}

fn fmt_stmts_and_call(
    cx: &LateContext<'_>,
    call_expr: &Expr<'_>,
    call_snippet: &str,
    args_snippets: Vec<Sugg<'_>>,
    non_empty_block_args_snippets: Vec<MaybeTypeUncertain<'_>>,
) -> String {
    let call_expr_indent = indent_of(cx, call_expr.span).unwrap_or(0);
    let call_snippet_with_replacements = args_snippets.into_iter().fold(call_snippet.to_owned(), |acc, arg| {
        acc.replacen(&arg.to_string(), "()", 1)
    });

    let stmts_and_call = non_empty_block_args_snippets
        .into_iter()
        .map(Into::into)
        .chain(iter::once(call_snippet_with_replacements))
        .map(|v| reindent_multiline(&v, true, Some(call_expr_indent)))
        .collect::<Vec<_>>();

    let mut stmts_and_call_snippet = stmts_and_call.join(&format!("{}{}", ";\n", " ".repeat(call_expr_indent)));
    // expr is not in a block statement or result expression position, wrap in a block
    let parent_node = cx.tcx.parent_hir_node(call_expr.hir_id);
    if !matches!(parent_node, Node::Block(_)) && !matches!(parent_node, Node::Stmt(_)) {
        let block_indent = call_expr_indent + 4;
        stmts_and_call_snippet = reindent_multiline(&stmts_and_call_snippet, true, Some(block_indent));
        stmts_and_call_snippet = format!(
            "{{\n{}{}\n{}}}",
            " ".repeat(block_indent),
            &stmts_and_call_snippet,
            " ".repeat(call_expr_indent)
        );
    }
    stmts_and_call_snippet
}
