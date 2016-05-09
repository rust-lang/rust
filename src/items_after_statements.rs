//! lint when items are used after statements

use rustc::lint::*;
use syntax::ast::*;
use utils::in_macro;

/// **What it does:** This lints checks for items declared after some statement in a block
///
/// **Why is this bad?** Items live for the entire scope they are declared in. But statements are
/// processed in order. This might cause confusion as it's hard to figure out which item is meant
/// in a statement.
///
/// **Known problems:** None
///
/// **Example:**
/// ```rust
/// fn foo() {
///     println!("cake");
/// }
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
    "finds blocks where an item comes after a statement"
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
        let mut stmts = item.stmts.iter().map(|stmt| &stmt.node);
        // skip initial items
        while let Some(&StmtKind::Decl(ref decl, _)) = stmts.next() {
            if let DeclKind::Local(_) = decl.node {
                break;
            }
        }
        // lint on all further items
        for stmt in stmts {
            if let StmtKind::Decl(ref decl, _) = *stmt {
                if let DeclKind::Item(ref it) = decl.node {
                    if in_macro(cx, it.span) {
                        return;
                    }
                    cx.struct_span_lint(ITEMS_AFTER_STATEMENTS,
                                        it.span,
                                        "adding items after statements is confusing, since items exist from the \
                                         start of the scope")
                      .emit();
                }
            }
        }
    }
}
