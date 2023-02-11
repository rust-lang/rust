// compile-flags: -Ztrait-solver=next
// edition: 2021
// revisions: pass fail
//[pass] check-pass

#![feature(generator_trait, generators)]

use std::ops::Generator;

struct A;
struct B;
struct C;

fn needs_generator(_: impl Generator<A, Yield = B, Return = C>) {}

#[cfg(fail)]
fn main() {
    needs_generator(|| {
        //[fail]~^ ERROR Generator<A>` is not satisfied
        //[fail]~| ERROR as Generator<A>>::Yield == B`
        //[fail]~| ERROR as Generator<A>>::Return == C`
        yield ();
    });
}

#[cfg(pass)]
fn main() {
    needs_generator(|_: A| {
        let _: A = yield B;
        C
    })
}
