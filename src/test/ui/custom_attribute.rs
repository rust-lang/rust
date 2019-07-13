#![feature(stmt_expr_attributes)]

#[foo] //~ ERROR the attribute `foo`
fn main() {
    #[foo] //~ ERROR the attribute `foo`
    let x = ();
    #[foo] //~ ERROR the attribute `foo`
    x
}
