#![feature(stmt_expr_attributes)]
#![deny(unused_parens)]

// Tests that lint attributes on statements/expressions are
// correctly applied to non-builtin early (AST) lints

fn main() {
    #[allow(unused_parens)]
    {
        let _ = (9);
    }
}
