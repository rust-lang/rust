// Regression test for #88653, where a confusing warning about a
// type mismatch in generator arguments was issued.

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn foo(bar: bool) -> impl Generator<(bool,)> {
//~^ ERROR: type mismatch in generator arguments [E0631]
//~| NOTE: expected signature of `fn((bool,)) -> _`
    |bar| {
    //~^ NOTE: found signature of `fn(bool) -> _`
        if bar {
            yield bar;
        }
    }
}

fn main() {}
