//@ compile-flags: -Znext-solver
//@ check-pass

// We must properly decrement available_depth when encountering cycles.
// Proving `Foo: Send` results in a cycle:
// - Foo: Send
// ->PhantomData<Foo>: Send
// ->Foo: Send
// and if we cache this result with too small of a required depth, the second
// `require` call will succeed with the cached result where it would fail if
// the goal were freshly evaluated.

#![recursion_limit = "6"]

use std::marker::PhantomData;

pub struct Foo(PhantomData<Foo>);
pub struct W<T>(T);

fn require<T: Send>() {}

fn main() {
    require::<Foo>();
    require::<W<W<W<W<W<Foo>>>>>>();
    //~^ WARN overflow evaluating the requirement `W<W<W<W<W<Foo>>>>>: Send`
    //~| WARN this was previously accepted by the compiler but is being phased out
}
