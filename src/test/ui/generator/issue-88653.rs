// Regression test for #88653, where a confusing warning about a
// type mismatch in generator arguments was issued.

#![feature(generators, generator_trait)]

use std::ops::Generator;

fn foo(bar: bool) -> impl Generator<(bool,)> {
    //~^ ERROR: type mismatch in generator arguments [E0631]
    //~| NOTE: expected due to this
    //~| NOTE: expected generator signature `fn((bool,)) -> _`
    //~| NOTE: in this expansion of desugaring of `impl Trait`
    //~| NOTE: in this expansion of desugaring of `impl Trait`
    |bar| {
        //~^ NOTE: found signature defined here
        if bar {
            yield bar;
        }
    }
}

fn main() {}
