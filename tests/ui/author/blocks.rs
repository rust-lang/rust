#![feature(stmt_expr_attributes)]

#[rustfmt::skip]
fn main() {
    #[clippy::author]
    {
        ;;;;
    }
}

#[clippy::author]
fn foo() {
    let x = 42i32;
    -x;
}
