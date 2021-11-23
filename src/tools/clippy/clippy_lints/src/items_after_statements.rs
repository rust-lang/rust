//! lint when items are used after statements

use clippy_utils::diagnostics::span_lint;
use rustc_ast::ast::{Block, ItemKind, StmtKind};
use rustc_lint::{EarlyContext, EarlyLintPass};
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
    /// ```rust
    /// // Bad
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
    /// ```rust
    /// // Good
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

impl EarlyLintPass for ItemsAfterStatements {
    fn check_block(&mut self, cx: &EarlyContext<'_>, item: &Block) {
        if in_external_macro(cx.sess, item.span) {
            return;
        }

        // skip initial items and trailing semicolons
        let stmts = item
            .stmts
            .iter()
            .map(|stmt| &stmt.kind)
            .skip_while(|s| matches!(**s, StmtKind::Item(..) | StmtKind::Empty));

        // lint on all further items
        for stmt in stmts {
            if let StmtKind::Item(ref it) = *stmt {
                if in_external_macro(cx.sess, it.span) {
                    return;
                }
                if let ItemKind::MacroDef(..) = it.kind {
                    // do not lint `macro_rules`, but continue processing further statements
                    continue;
                }
                span_lint(
                    cx,
                    ITEMS_AFTER_STATEMENTS,
                    it.span,
                    "adding items after statements is confusing, since items exist from the \
                     start of the scope",
                );
            }
        }
    }
}
