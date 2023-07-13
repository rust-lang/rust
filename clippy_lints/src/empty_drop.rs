use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::peel_blocks;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Body, ExprKind, Impl, ImplItemKind, Item, ItemKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty `Drop` implementations.
    ///
    /// ### Why is this bad?
    /// Empty `Drop` implementations have no effect when dropping an instance of the type. They are
    /// most likely useless. However, an empty `Drop` implementation prevents a type from being
    /// destructured, which might be the intention behind adding the implementation as a marker.
    ///
    /// ### Example
    /// ```rust
    /// struct S;
    ///
    /// impl Drop for S {
    ///     fn drop(&mut self) {}
    /// }
    /// ```
    /// Use instead:
    /// ```rust
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
        if_chain! {
            if let ItemKind::Impl(Impl {
                of_trait: Some(ref trait_ref),
                items: [child],
                ..
            }) = item.kind;
            if trait_ref.trait_def_id() == cx.tcx.lang_items().drop_trait();
            if let impl_item_hir = child.id.hir_id();
            if let Some(Node::ImplItem(impl_item)) = cx.tcx.hir().find(impl_item_hir);
            if let ImplItemKind::Fn(_, b) = &impl_item.kind;
            if let Body { value: func_expr, .. } = cx.tcx.hir().body(*b);
            let func_expr = peel_blocks(func_expr);
            if let ExprKind::Block(block, _) = func_expr.kind;
            if block.stmts.is_empty() && block.expr.is_none();
            then {
                span_lint_and_sugg(
                    cx,
                    EMPTY_DROP,
                    item.span,
                    "empty drop implementation",
                    "try removing this impl",
                    String::new(),
                    Applicability::MaybeIncorrect
                );
            }
        }
    }
}
