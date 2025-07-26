use clippy_utils::macros::{is_panic, macro_backtrace};
use rustc_ast::{AttrId, MetaItemInner};
use rustc_hir::{
    Block, Expr, ExprKind, ImplItem, ImplItemKind, Item, ItemKind, StmtKind, TraitFn, TraitItem, TraitItemKind,
};
use rustc_lint::{LateContext, Level};
use rustc_middle::ty;
use rustc_span::sym;
use rustc_span::symbol::Symbol;

pub(super) fn is_word(nmi: &MetaItemInner, expected: Symbol) -> bool {
    if let MetaItemInner::MetaItem(mi) = &nmi {
        mi.is_word() && mi.has_name(expected)
    } else {
        false
    }
}

pub(super) fn is_lint_level(symbol: Symbol, attr_id: AttrId) -> bool {
    Level::from_symbol(symbol, || Some(attr_id)).is_some()
}

pub(super) fn is_relevant_item(cx: &LateContext<'_>, item: &Item<'_>) -> bool {
    if let ItemKind::Fn { body: eid, .. } = item.kind {
        is_relevant_expr(cx, cx.tcx.typeck_body(eid), cx.tcx.hir_body(eid).value)
    } else {
        false
    }
}

pub(super) fn is_relevant_impl(cx: &LateContext<'_>, item: &ImplItem<'_>) -> bool {
    match item.kind {
        ImplItemKind::Fn(_, eid) => is_relevant_expr(cx, cx.tcx.typeck_body(eid), cx.tcx.hir_body(eid).value),
        _ => false,
    }
}

pub(super) fn is_relevant_trait(cx: &LateContext<'_>, item: &TraitItem<'_>) -> bool {
    match item.kind {
        TraitItemKind::Fn(_, TraitFn::Required(_)) => true,
        TraitItemKind::Fn(_, TraitFn::Provided(eid)) => {
            is_relevant_expr(cx, cx.tcx.typeck_body(eid), cx.tcx.hir_body(eid).value)
        },
        _ => false,
    }
}

fn is_relevant_block(cx: &LateContext<'_>, typeck_results: &ty::TypeckResults<'_>, block: &Block<'_>) -> bool {
    block.stmts.first().map_or_else(
        || {
            block
                .expr
                .as_ref()
                .is_some_and(|e| is_relevant_expr(cx, typeck_results, e))
        },
        |stmt| match &stmt.kind {
            StmtKind::Let(_) => true,
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => is_relevant_expr(cx, typeck_results, expr),
            StmtKind::Item(_) => false,
        },
    )
}

fn is_relevant_expr(cx: &LateContext<'_>, typeck_results: &ty::TypeckResults<'_>, expr: &Expr<'_>) -> bool {
    if macro_backtrace(expr.span).last().is_some_and(|macro_call| {
        is_panic(cx, macro_call.def_id) || cx.tcx.item_name(macro_call.def_id) == sym::unreachable
    }) {
        return false;
    }
    match &expr.kind {
        ExprKind::Block(block, _) => is_relevant_block(cx, typeck_results, block),
        ExprKind::Ret(Some(e)) => is_relevant_expr(cx, typeck_results, e),
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
