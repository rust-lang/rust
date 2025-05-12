//@ compile-flags: -Zcrate-attr=feature(yeet_expr)]#![allow(warnings)
//~? ERROR unexpected token
fn foo() {} //~ ERROR `main` function not found
