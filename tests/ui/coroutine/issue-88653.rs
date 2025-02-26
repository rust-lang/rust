// Regression test for #88653, where a confusing warning about a
// type mismatch in coroutine arguments was issued.

#![feature(coroutines, coroutine_trait)]

use std::ops::Coroutine;

fn foo(bar: bool) -> impl Coroutine<(bool,)> {
    //~^ ERROR: type mismatch in coroutine arguments [E0631]
    //~| NOTE: expected due to this
    //~| NOTE: expected coroutine signature `fn((bool,)) -> _`
    //~| NOTE: in this expansion of desugaring of `impl Trait`
    //~| NOTE: in this expansion of desugaring of `impl Trait`
    #[coroutine]
    |bar| {
        //~^ NOTE: found signature defined here
        //~| NOTE: return type was inferred to be
        if bar {
            yield bar;
        }
    }
}

fn main() {}
