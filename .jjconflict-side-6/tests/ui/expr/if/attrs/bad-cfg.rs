#![feature(stmt_expr_attributes)]

fn main() {
    let _ = #[cfg(false)] if true {}; //~ ERROR removing an expression
}
