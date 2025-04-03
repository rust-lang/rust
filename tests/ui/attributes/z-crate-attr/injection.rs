//@ compile-flags: '-Zcrate-attr=feature(yeet_expr)]fn main(){}#[inline'
//@ error-pattern:unexpected token
fn foo() {}
