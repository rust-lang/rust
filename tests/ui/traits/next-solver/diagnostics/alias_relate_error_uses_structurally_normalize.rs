//@ compile-flags: -Znext-solver

// When encountering a fulfillment error from an `alias-relate` goal failing, we
// would previously manually construct a `normalizes-to` goal involving the alias
// and an infer var. This would then ICE as normalization would return a nested
// goal (the `T: Sized` from the `Trait` impl for `Foo<T>` below) from the root goal
// which is not supported.

struct Foo<T>(T);

trait Trait {
    type Assoc;
}

// `T: Sized` being explicit is not required, but the bound being present *is*.
impl<T: Sized> Trait for Foo<T> {
    type Assoc = u64;
}

fn bar<T: Trait<Assoc = u32>>(_: T) {}

fn main() {
    let foo = Foo(Default::default());
    bar(foo);
    //~^ ERROR: type mismatch resolving `<Foo<_> as Trait>::Assoc == u32`
    // Here diagnostics would manually construct a `<Foo<?y> as Trait>::Assoc normalizes-to ?x` goal
    // which would return a nested goal of `?y: Sized` from the impl.
}
