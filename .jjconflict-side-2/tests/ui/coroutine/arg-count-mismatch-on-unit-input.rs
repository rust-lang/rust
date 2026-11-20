#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;

fn foo() -> impl Coroutine<u8> {
    //~^ ERROR type mismatch in coroutine arguments
    #[coroutine]
    |_: ()| {}
}

fn main() { }
