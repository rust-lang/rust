// revisions: e2024 none
//[e2024] compile-flags: --edition 2024 -Zunstable-options

fn main() {
    let x = gen {};
    //[none]~^ ERROR: cannot find
    //[e2024]~^^ ERROR: `gen` blocks are not yet implemented
    let y = gen { yield 42 };
    //[none]~^ ERROR: found reserved keyword `yield`
    //[none]~| ERROR: cannot find
    gen {};
    //[none]~^ ERROR: cannot find
}
