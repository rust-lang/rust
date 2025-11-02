use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::peel_blocks;
use rustc_errors::Applicability;
use rustc_hir::{Body, ExprKind, Impl, ImplItemKind, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty `Drop` implementations.
    ///
    /// ### Why restrict this?
    /// Empty `Drop` implementations have no effect when dropping an instance of the type. They are
    /// most likely useless. However, an empty `Drop` implementation prevents a type from being
    /// destructured, which might be the intention behind adding the implementation as a marker.
    ///
    /// ### Example
    /// ```no_run
    /// struct S;
    ///
    /// impl Drop for S {
    ///     fn drop(&mut self) {}
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct S;
    /// ```
    #[clippy::version = "1.62.0"]
    pub EMPTY_DROP,
    restriction,
    "empty `Drop` implementations"
}
declare_lint_pass!(EmptyDrop => [EMPTY_DROP]);

impl LateLintPass<'_> for EmptyDrop {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if let ItemKind::Impl(Impl {
            of_trait: Some(of_trait),
            items: [child],
            ..
        }) = item.kind
            && of_trait.trait_ref.trait_def_id() == cx.tcx.lang_items().drop_trait()
            && let impl_item_hir = child.hir_id()
            && let Node::ImplItem(impl_item) = cx.tcx.hir_node(impl_item_hir)
            && let ImplItemKind::Fn(_, b) = &impl_item.kind
            && let Body { value: func_expr, .. } = cx.tcx.hir_body(*b)
            && let func_expr = peel_blocks(func_expr)
            && let ExprKind::Block(block, _) = func_expr.kind
            && block.stmts.is_empty()
            && block.expr.is_none()
        {
            span_lint_and_then(cx, EMPTY_DROP, item.span, "empty drop implementation", |diag| {
                diag.span_suggestion_hidden(
                    item.span,
                    "try removing this impl",
                    String::new(),
                    Applicability::MaybeIncorrect,
                );
            });
        }
    }
}
