//! lint when items are used after statements

use rustc::lint::*;
use syntax::ast::*;
use utils::{in_macro, span_lint};

/// **What it does:** Checks for items declared after some statement in a block.
///
/// **Why is this bad?** Items live for the entire scope they are declared
/// in. But statements are processed in order. This might cause confusion as
/// it's hard to figure out which item is meant in a statement.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
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
declare_lint! {
    pub ITEMS_AFTER_STATEMENTS,
    Allow,
    "blocks where an item comes after a statement"
}

pub struct ItemsAfterStatements;

impl LintPass for ItemsAfterStatements {
    fn get_lints(&self) -> LintArray {
        lint_array!(ITEMS_AFTER_STATEMENTS)
    }
}

impl EarlyLintPass for ItemsAfterStatements {
    fn check_block(&mut self, cx: &EarlyContext, item: &Block) {
        if in_macro(cx, item.span) {
            return;
        }

        // skip initial items
        let stmts = item.stmts
            .iter()
            .map(|stmt| &stmt.node)
            .skip_while(|s| matches!(**s, StmtKind::Item(..)));

        // lint on all further items
        for stmt in stmts {
            if let StmtKind::Item(ref it) = *stmt {
                if in_macro(cx, it.span) {
                    return;
                }
                span_lint(cx,
                          ITEMS_AFTER_STATEMENTS,
                          it.span,
                          "adding items after statements is confusing, since items exist from the \
                           start of the scope");
            }
        }
    }
}
