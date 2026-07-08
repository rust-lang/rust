use clippy_utils::macros::{is_panic, macro_backtrace};
use rustc_ast::MetaItemInner;
use rustc_hir::{Block, Expr, ExprKind, StmtKind};
use rustc_lint::{LateContext, Level};
use rustc_span::sym;
use rustc_span::symbol::Symbol;

pub(super) fn is_word(nmi: &MetaItemInner, expected: Symbol) -> bool {
    if let MetaItemInner::MetaItem(mi) = &nmi {
        mi.is_word() && mi.has_name(expected)
    } else {
        false
    }
}

pub(super) fn is_lint_level(symbol: Symbol) -> bool {
    Level::from_symbol(symbol).is_some()
}

fn is_relevant_block(cx: &LateContext<'_>, block: &Block<'_>) -> bool {
    block.stmts.first().map_or_else(
        || block.expr.as_ref().is_some_and(|e| is_relevant_expr(cx, e)),
        |stmt| match &stmt.kind {
            StmtKind::Let(_) => true,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => is_relevant_expr(cx, expr),
            StmtKind::Item(_) => false,
        },
    )
}

pub(super) fn is_relevant_expr(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if macro_backtrace(expr.span).last().is_some_and(|macro_call| {
        is_panic(cx, macro_call.def_id) || cx.tcx.item_name(macro_call.def_id) == sym::unreachable
    }) {
        return false;
    }
    match &expr.kind {
        ExprKind::Block(block, _) => is_relevant_block(cx, block),
        ExprKind::Ret(Some(e)) => is_relevant_expr(cx, e),
        ExprKind::Ret(None) | ExprKind::Break(_, None) => false,
        _ => true,
    }
}

/// Returns the lint name if it is clippy lint.
pub(super) fn extract_clippy_lint(lint: &MetaItemInner) -> Option<Symbol> {
    match namespace_and_lint(lint) {
        (Some(sym::clippy), name) => name,
        _ => None,
    }
}

/// Returns the lint namespace, if any, as well as the lint name. (`None`, `None`) means
/// the lint had less than 1 or more than 2 segments.
pub(super) fn namespace_and_lint(lint: &MetaItemInner) -> (Option<Symbol>, Option<Symbol>) {
    match lint.meta_item().map(|m| m.path.segments.as_slice()).unwrap_or_default() {
        [name] => (None, Some(name.ident.name)),
        [namespace, name] => (Some(namespace.ident.name), Some(name.ident.name)),
        _ => (None, None),
    }
}
