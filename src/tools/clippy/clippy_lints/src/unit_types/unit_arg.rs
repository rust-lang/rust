use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_from_proc_macro;
use clippy_utils::source::{indent_of, reindent_multiline, snippet_opt};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, Block, Expr, ExprKind, MatchSource, Node, StmtKind};
use rustc_lint::LateContext;

use super::{utils, UNIT_ARG};

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
    let map = &cx.tcx.hir();
    let opt_parent_node = map.find(map.get_parent_node(expr.hir_id));
    if_chain! {
        if let Some(hir::Node::Expr(parent_expr)) = opt_parent_node;
        if is_questionmark_desugar_marked_call(parent_expr);
        then {
            return;
        }
    }

    let args: Vec<_> = match expr.kind {
        ExprKind::Call(_, args) => args.iter().collect(),
        ExprKind::MethodCall(_, receiver, args, _) => std::iter::once(receiver).chain(args.iter()).collect(),
        _ => return,
    };

    let args_to_recover = args
        .into_iter()
        .filter(|arg| {
            if cx.typeck_results().expr_ty(arg).is_unit() && !utils::is_unit_literal(arg) {
                !matches!(
                    &arg.kind,
                    ExprKind::Match(.., MatchSource::TryDesugar) | ExprKind::Path(..)
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

fn lint_unit_args(cx: &LateContext<'_>, expr: &Expr<'_>, args_to_recover: &[&Expr<'_>]) {
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
        &format!("passing {}unit value{} to a function", singular, plural),
        |db| {
            let mut or = "";
            args_to_recover
                .iter()
                .filter_map(|arg| {
                    if_chain! {
                        if let ExprKind::Block(block, _) = arg.kind;
                        if block.expr.is_none();
                        if let Some(last_stmt) = block.stmts.iter().last();
                        if let StmtKind::Semi(last_expr) = last_stmt.kind;
                        if let Some(snip) = snippet_opt(cx, last_expr.span);
                        then {
                            Some((
                                last_stmt.span,
                                snip,
                            ))
                        }
                        else {
                            None
                        }
                    }
                })
                .for_each(|(span, sugg)| {
                    db.span_suggestion(
                        span,
                        "remove the semicolon from the last statement in the block",
                        sugg,
                        Applicability::MaybeIncorrect,
                    );
                    or = "or ";
                    applicability = Applicability::MaybeIncorrect;
                });

            let arg_snippets: Vec<String> = args_to_recover
                .iter()
                .filter_map(|arg| snippet_opt(cx, arg.span))
                .collect();
            let arg_snippets_without_empty_blocks: Vec<String> = args_to_recover
                .iter()
                .filter(|arg| !is_empty_block(arg))
                .filter_map(|arg| snippet_opt(cx, arg.span))
                .collect();

            if let Some(call_snippet) = snippet_opt(cx, expr.span) {
                let sugg = fmt_stmts_and_call(
                    cx,
                    expr,
                    &call_snippet,
                    &arg_snippets,
                    &arg_snippets_without_empty_blocks,
                );

                if arg_snippets_without_empty_blocks.is_empty() {
                    db.multipart_suggestion(
                        &format!("use {}unit literal{} instead", singular, plural),
                        args_to_recover
                            .iter()
                            .map(|arg| (arg.span, "()".to_string()))
                            .collect::<Vec<_>>(),
                        applicability,
                    );
                } else {
                    let plural = arg_snippets_without_empty_blocks.len() > 1;
                    let empty_or_s = if plural { "s" } else { "" };
                    let it_or_them = if plural { "them" } else { "it" };
                    db.span_suggestion(
                        expr.span,
                        &format!(
                            "{}move the expression{} in front of the call and replace {} with the unit literal `()`",
                            or, empty_or_s, it_or_them
                        ),
                        sugg,
                        applicability,
                    );
                }
            }
        },
    );
}

fn is_empty_block(expr: &Expr<'_>) -> bool {
    matches!(
        expr.kind,
        ExprKind::Block(
            Block {
                stmts: &[],
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
    args_snippets: &[impl AsRef<str>],
    non_empty_block_args_snippets: &[impl AsRef<str>],
) -> String {
    let call_expr_indent = indent_of(cx, call_expr.span).unwrap_or(0);
    let call_snippet_with_replacements = args_snippets
        .iter()
        .fold(call_snippet.to_owned(), |acc, arg| acc.replacen(arg.as_ref(), "()", 1));

    let mut stmts_and_call = non_empty_block_args_snippets
        .iter()
        .map(|it| it.as_ref().to_owned())
        .collect::<Vec<_>>();
    stmts_and_call.push(call_snippet_with_replacements);
    stmts_and_call = stmts_and_call
        .into_iter()
        .map(|v| reindent_multiline(v.into(), true, Some(call_expr_indent)).into_owned())
        .collect();

    let mut stmts_and_call_snippet = stmts_and_call.join(&format!("{}{}", ";\n", " ".repeat(call_expr_indent)));
    // expr is not in a block statement or result expression position, wrap in a block
    let parent_node = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(call_expr.hir_id));
    if !matches!(parent_node, Some(Node::Block(_))) && !matches!(parent_node, Some(Node::Stmt(_))) {
        let block_indent = call_expr_indent + 4;
        stmts_and_call_snippet =
            reindent_multiline(stmts_and_call_snippet.into(), true, Some(block_indent)).into_owned();
        stmts_and_call_snippet = format!(
            "{{\n{}{}\n{}}}",
            " ".repeat(block_indent),
            &stmts_and_call_snippet,
            " ".repeat(call_expr_indent)
        );
    }
    stmts_and_call_snippet
}
