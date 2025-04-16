#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn fine(x: impl Into<u32>) -> impl Into<u32> { x }

fn bad_in_ret_position(x: impl Into<u32>) -> impl Into<impl Debug> { x }
//~^ ERROR nested `impl Trait` is not allowed
//~| ERROR the trait bound `impl Debug: From<impl Into<u32>>` is not satisfied

fn bad_in_fn_syntax(x: fn() -> impl Into<impl Debug>) {}
//~^ ERROR nested `impl Trait` is not allowed
//~| ERROR `impl Trait` is not allowed in `fn` pointer

fn bad_in_arg_position(_: impl Into<impl Debug>) { }
//~^ ERROR nested `impl Trait` is not allowed

struct X;
impl X {
    fn bad(x: impl Into<u32>) -> impl Into<impl Debug> { x }
    //~^ ERROR nested `impl Trait` is not allowed
    //~| ERROR the trait bound `impl Debug: From<impl Into<u32>>` is not satisfied
}

fn allowed_in_assoc_type() -> impl Iterator<Item=impl Fn()> {
    vec![|| println!("woot")].into_iter()
}

fn allowed_in_ret_type() -> impl Fn() -> impl Into<u32> {
    || 5u8
}

fn main() {}
