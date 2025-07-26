use super::WHILE_LET_LOOP;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{snippet, snippet_indent, snippet_opt};
use clippy_utils::ty::needs_ordered_drop;
use clippy_utils::visitors::any_temporaries_need_ordered_drop;
use clippy_utils::{higher, peel_blocks};
use rustc_ast::BindingMode;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, LetStmt, MatchSource, Pat, PatKind, Path, QPath, StmtKind, Ty};
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, loop_block: &'tcx Block<'_>) {
    let (init, let_info) = match (loop_block.stmts, loop_block.expr) {
        ([stmt, ..], _) => match stmt.kind {
            StmtKind::Let(LetStmt {
                init: Some(e),
                els: None,
                pat,
                ty,
                ..
            }) => (*e, Some((*pat, *ty))),
            StmtKind::Semi(e) | StmtKind::Expr(e) => (e, None),
            _ => return,
        },
        ([], Some(e)) => (e, None),
        _ => return,
    };
    let has_trailing_exprs = loop_block.stmts.len() + usize::from(loop_block.expr.is_some()) > 1;

    if let Some(if_let) = higher::IfLet::hir(cx, init)
        && let Some(else_expr) = if_let.if_else
        && is_simple_break_expr(else_expr)
    {
        could_be_while_let(
            cx,
            expr,
            if_let.let_pat,
            if_let.let_expr,
            has_trailing_exprs,
            let_info,
            if_let.if_then,
        );
    } else if let ExprKind::Match(scrutinee, [arm1, arm2], MatchSource::Normal) = init.kind
        && arm1.guard.is_none()
        && arm2.guard.is_none()
        && is_simple_break_expr(arm2.body)
    {
        could_be_while_let(cx, expr, arm1.pat, scrutinee, has_trailing_exprs, let_info, arm1.body);
    }
}

/// Returns `true` if expr contains a single break expression without a label or sub-expression,
/// possibly embedded in blocks.
fn is_simple_break_expr(e: &Expr<'_>) -> bool {
    if let ExprKind::Block(b, _) = e.kind {
        match (b.stmts, b.expr) {
            ([s], None) => matches!(s.kind, StmtKind::Expr(e) | StmtKind::Semi(e) if is_simple_break_expr(e)),
            ([], Some(e)) => is_simple_break_expr(e),
            _ => false,
        }
    } else {
        matches!(e.kind, ExprKind::Break(dest, None) if dest.label.is_none())
    }
}

fn could_be_while_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    has_trailing_exprs: bool,
    let_info: Option<(&Pat<'_>, Option<&Ty<'_>>)>,
    inner_expr: &Expr<'_>,
) {
    if has_trailing_exprs
        && (needs_ordered_drop(cx, cx.typeck_results().expr_ty(let_expr))
            || any_temporaries_need_ordered_drop(cx, let_expr))
    {
        // Switching to a `while let` loop will extend the lifetime of some values.
        return;
    }

    // NOTE: we used to build a body here instead of using
    // ellipsis, this was removed because:
    // 1) it was ugly with big bodies;
    // 2) it was not indented properly;
    // 3) it wasn’t very smart (see #675).
    let inner_content = if let Some((pat, ty)) = let_info
        // Prevent trivial reassignments such as `let x = x;` or `let _ = …;`, but
        // keep them if the type has been explicitly specified.
        && (!is_trivial_assignment(pat, peel_blocks(inner_expr)) || ty.is_some())
        && let Some(pat_str) = snippet_opt(cx, pat.span)
        && let Some(init_str) = snippet_opt(cx, peel_blocks(inner_expr).span)
    {
        let ty_str = ty
            .map(|ty| format!(": {}", snippet(cx, ty.span, "_")))
            .unwrap_or_default();
        format!(
            "\n{indent}    let {pat_str}{ty_str} = {init_str};\n{indent}    ..\n{indent}",
            indent = snippet_indent(cx, expr.span).unwrap_or_default(),
        )
    } else {
        " .. ".into()
    };

    span_lint_and_sugg(
        cx,
        WHILE_LET_LOOP,
        expr.span,
        "this loop could be written as a `while let` loop",
        "try",
        format!(
            "while let {} = {} {{{inner_content}}}",
            snippet(cx, let_pat.span, ".."),
            snippet(cx, let_expr.span, ".."),
        ),
        Applicability::HasPlaceholders,
    );
}

fn is_trivial_assignment(pat: &Pat<'_>, init: &Expr<'_>) -> bool {
    match (pat.kind, init.kind) {
        (PatKind::Wild, _) => true,
        (
            PatKind::Binding(BindingMode::NONE, _, pat_ident, None),
            ExprKind::Path(QPath::Resolved(None, Path { segments: [init], .. })),
        ) => pat_ident.name == init.ident.name,
        _ => false,
    }
}
