//@ compile-flags: -Znext-solver
//@ check-pass

#![recursion_limit = "6"]
use std::marker::PhantomData;

// We need to decrement available_depth on provisional cache hits.
// e.g., in the following example:
// A: Send, avail = 6
// -> PhantomData<A>: Send, avail = 5
//    -> A: Send, avail = 4
// -> B: Send, avail = 5
//    -> C: Send, avail = 4
//       -> PhantomData<A>: Send, provisional cache hit, avail = 4 if we don't decrement
struct A(PhantomData<A>, B);
struct B(C);
struct C(PhantomData<A>);
struct W<T>(T);

fn require<T: Send>() {}

fn main() {
    require::<A>();
    require::<W<W<W<W<A>>>>>();
    //~^ WARN overflow evaluating the requirement `W<W<W<W<A>>>>: Send`
    //~| WARN this was previously accepted by the compiler but is being phased out
}
