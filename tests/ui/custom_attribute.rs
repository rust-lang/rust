#![feature(stmt_expr_attributes)]

#[foo] //~ ERROR cannot find attribute `foo`
fn main() {
    #[foo] //~ ERROR cannot find attribute `foo`
    let x = ();
    #[foo] //~ ERROR cannot find attribute `foo`
    x
}
