//! Regression test for <https://github.com/rust-lang/rust/issues/109864>.
//!
//! Deeply nested associated-type projections used to ICE with "type variables
//! should not be hashed" — but only under incremental compilation. They should now
//! produce ordinary trait-bound errors instead of crashing.

//@ incremental

struct S;
struct S2<P>(); //~ ERROR type parameter `P` is never used

trait Foo<A> {
    type Out;
}

trait Bar<A, B> {
    type Out;
}

trait Qux<A> {
    type Out;
}

trait Fuzz<A> {
    type Out;
}

impl<A: Foo<B>, B> Fuzz<S2<B>> for S2<A> {
    type Out = <<A as Foo<B>>::Out as Bar<
        //~^ ERROR the trait bound `<A as Foo<B>>::Out: Qux<S>` is not satisfied
        //~| ERROR the trait bound `<A as Foo<B>>::Out: Bar<S2<A>, _>` is not satisfied
        //~| ERROR the trait bound `<A as Foo<B>>::Out: Bar<S2<A>, _>` is not satisfied
        S2<A>,
        <<<A as Foo<B>>::Out as Qux<S>>::Out as Fuzz<S2<B>>>::Out,
        //~^ ERROR the trait bound `<A as Foo<B>>::Out: Qux<S>` is not satisfied
    >>::Out;
}

fn main() {}
