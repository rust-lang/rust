//@ revisions: e2024 none
//@[e2024] edition: 2024
#![cfg_attr(e2024, feature(gen_blocks))]
#![feature(stmt_expr_attributes)]

fn main() {
    let x = gen {};
    //[none]~^ ERROR: cannot find
    //[e2024]~^^ ERROR: type annotations needed
    let y = gen { yield 42 };
    //[none]~^ ERROR: found reserved keyword `yield`
    //[none]~| ERROR: cannot find
    gen {};
    //[none]~^ ERROR: cannot find

    let _ = || yield true; //[none]~ ERROR yield syntax is experimental
    //[none]~^ ERROR yield syntax is experimental
    //~^^ ERROR `yield` can only be used in

    let _ = #[coroutine] || yield true; //[none]~ ERROR yield syntax is experimental
    //~^ ERROR `#[coroutine]` attribute is an experimental feature
    //[none]~^^ ERROR yield syntax is experimental

    let _ = #[coroutine] || {};
    //~^ ERROR `#[coroutine]` attribute is an experimental feature
}
