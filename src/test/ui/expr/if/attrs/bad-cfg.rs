#![feature(stmt_expr_attributes)]

fn main() {
    let _ = #[cfg(FALSE)] if true {}; //~ ERROR removing an expression
}
