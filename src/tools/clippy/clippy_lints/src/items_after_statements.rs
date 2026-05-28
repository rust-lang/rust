use clippy_utils::diagnostics::span_lint_hir;
use rustc_hir::{Block, ItemKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for items declared after some statement in a block.
    ///
    /// ### Why is this bad?
    /// Items live for the entire scope they are declared
    /// in. But statements are processed in order. This might cause confusion as
    /// it's hard to figure out which item is meant in a statement.
    ///
    /// ### Example
    /// ```no_run
    /// fn foo() {
    ///     println!("cake");
    /// }
    ///
    /// fn main() {
    ///     foo(); // prints "foo"
    ///     fn foo() {
    ///         println!("foo");
    ///     }
    ///     foo(); // prints "foo"
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn foo() {
    ///     println!("cake");
    /// }
    ///
    /// fn main() {
    ///     fn foo() {
    ///         println!("foo");
    ///     }
    ///     foo(); // prints "foo"
    ///     foo(); // prints "foo"
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ITEMS_AFTER_STATEMENTS,
    pedantic,
    "blocks where an item comes after a statement"
}

declare_lint_pass!(ItemsAfterStatements => [ITEMS_AFTER_STATEMENTS]);

impl LateLintPass<'_> for ItemsAfterStatements {
    fn check_block(&mut self, cx: &LateContext<'_>, block: &Block<'_>) {
        if block.stmts.len() > 1 {
            let ctxt = block.span.ctxt();
            let mut in_external = None;
            block
                .stmts
                .iter()
                .skip_while(|stmt| matches!(stmt.kind, StmtKind::Item(..)))
                .filter_map(|stmt| match stmt.kind {
                    StmtKind::Item(id) => Some(cx.tcx.hir_item(id)),
                    _ => None,
                })
                // Ignore macros since they can only see previously defined locals.
                .filter(|item| !matches!(item.kind, ItemKind::Macro(..)))
                // Stop linting if macros define items.
                .take_while(|item| item.span.ctxt() == ctxt)
                // Don't use `next` due to the complex filter chain.
                .for_each(|item| {
                    // Only do the macro check once, but delay it until it's needed.
                    if !*in_external.get_or_insert_with(|| block.span.in_external_macro(cx.sess().source_map())) {
                        span_lint_hir(
                            cx,
                            ITEMS_AFTER_STATEMENTS,
                            item.hir_id(),
                            item.span,
                            "adding items after statements is confusing, since items exist from the \
                                start of the scope",
                        );
                    }
                });
        }
    }
}
