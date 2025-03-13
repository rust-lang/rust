//@ compile-flags: '-Zcrate-attr=feature(yeet_expr)]fn main(){}#[inline'
//@ error-pattern:unexpected closing delimiter
fn foo() {}
