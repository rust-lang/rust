//@ compile-flags: '-Zcrate-attr=feature(yeet_expr)]fn main(){}#[inline'

fn foo() {}

//~? ERROR unexpected closing delimiter: `]`
