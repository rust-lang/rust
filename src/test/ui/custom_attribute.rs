#![feature(stmt_expr_attributes)]

#[foo] //~ ERROR cannot find attribute macro `foo` in this scope
fn main() {
    #[foo] //~ ERROR cannot find attribute macro `foo` in this scope
    let x = ();
    #[foo] //~ ERROR cannot find attribute macro `foo` in this scope
    x
}
