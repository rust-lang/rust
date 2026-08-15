//@ compile-flags: -Znext-solver --diagnostic-width=300
//@ edition: 2021
//@ revisions: pass fail
//@[pass] check-pass

#![feature(coroutine_trait, coroutines)]

use std::ops::Coroutine;

struct A;
struct B;
struct C;

fn needs_coroutine(_: impl Coroutine<A, Yield = B, Return = C>) {}

#[cfg(fail)]
fn main() {
    needs_coroutine(
        #[coroutine]
        || {
            //[fail]~^ ERROR Coroutine<A>` is not satisfied
            yield ();
        },
    );
}

#[cfg(pass)]
fn main() {
    needs_coroutine(
        #[coroutine]
        |_: A| {
            let _: A = yield B;
            C
        },
    )
}
