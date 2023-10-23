//! lint when items are used after statements

use clippy_utils::diagnostics::span_lint_hir;
use rustc_hir::{Block, ItemKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
        if in_external_macro(cx.sess(), block.span) {
            return;
        }

        // skip initial items
        let stmts = block
            .stmts
            .iter()
            .skip_while(|stmt| matches!(stmt.kind, StmtKind::Item(..)));

        // lint on all further items
        for stmt in stmts {
            if let StmtKind::Item(item_id) = stmt.kind {
                let item = cx.tcx.hir().item(item_id);
                if in_external_macro(cx.sess(), item.span) || !item.span.eq_ctxt(block.span) {
                    return;
                }
                if let ItemKind::Macro(..) = item.kind {
                    // do not lint `macro_rules`, but continue processing further statements
                    continue;
                }
                span_lint_hir(
                    cx,
                    ITEMS_AFTER_STATEMENTS,
                    item.hir_id(),
                    item.span,
                    "adding items after statements is confusing, since items exist from the \
                     start of the scope",
                );
            }
        }
    }
}
