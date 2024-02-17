//@ revisions: e2024 none
//@[e2024] compile-flags: --edition 2024 -Zunstable-options
#![cfg_attr(e2024, feature(gen_blocks))]

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
    //~^ ERROR yield syntax is experimental
}
