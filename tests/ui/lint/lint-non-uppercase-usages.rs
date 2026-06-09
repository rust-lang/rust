// Checks that the `non_upper_case_globals` emits suggestions for usages as well
// <https://github.com/rust-lang/rust/issues/124061>

//@ check-pass
//@ run-rustfix

#![allow(dead_code)]

use std::cell::Cell;

const my_static: u32 = 0;
//~^ WARN constant `my_static` should have an upper case name
//~| SUGGESTION MY_STATIC

const LOL: u32 = my_static + 0;
//~^ SUGGESTION MY_STATIC

mod my_mod {
    const INSIDE_MOD: u32 = super::my_static + 0;
    //~^ SUGGESTION MY_STATIC
}

thread_local! {
    static fooFOO: Cell<usize> = unreachable!();
    //~^ WARN constant `fooFOO` should have an upper case name
    //~| SUGGESTION FOO_FOO
}

fn foo<const foo: u32>() {
    //~^ WARN const parameter `foo` should have an upper case name
    //~| SUGGESTION FOO
    let _a = foo + 1;
    //~^ SUGGESTION FOO
}

fn main() {
    let _a = crate::my_static;
    //~^ SUGGESTION MY_STATIC

    fooFOO.set(9);
    //~^ SUGGESTION FOO_FOO
    println!("{}", fooFOO.get());
    //~^ SUGGESTION FOO_FOO
}
