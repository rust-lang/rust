#![feature(stmt_expr_attributes)]

#[foo] //~ ERROR The attribute `foo`
fn main() {
    #[foo] //~ ERROR The attribute `foo`
    let x = ();
    #[foo] //~ ERROR The attribute `foo`
    x
}
