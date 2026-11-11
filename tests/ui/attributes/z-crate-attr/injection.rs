//@ compile-flags: '-Zcrate-attr=feature(yeet_expr)]fn main(){}#[inline'
//~? ERROR unexpected token
fn foo() {} //~ ERROR `main` function not found
