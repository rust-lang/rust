//! Checks error handling for undefined custom attributes.

#![feature(stmt_expr_attributes)]

#[foo] //~ ERROR cannot find attribute `foo` in this scope
fn main() {
    #[foo] //~ ERROR cannot find attribute `foo` in this scope
    let x = ();
    #[foo] //~ ERROR cannot find attribute `foo` in this scope
    x
}
