#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

// Test whether IAT resolution in item signatures will actually instantiate the
// impl's params with infers before equating self types, or if we "cheat" and
// use a heuristic (e.g. DeepRejectCtxt).

struct Foo<T, U, V>(T, U, V);

impl<T> Foo<T, T, u8> {
    type IAT = u8;
}

impl<T, U> Foo<T, U, u16> {
    type IAT = u16;
}

trait Identity {
    type This;
}
impl<T> Identity for T {
    type This = T;
}

struct Bar<T, U> {
    // It would be illegal to resolve to `Foo<T, T, u8>::IAT`  as  `T` and `U` are
    // different types. However, currently we treat all impl-side params sort of like
    // they're infers and assume they can unify with anything, so we consider it a
    // valid candidate.
    field: Foo<T, U, <u16 as Identity>::This>::IAT,
    //~^ ERROR: multiple applicable items in scope
}

fn main() {}
