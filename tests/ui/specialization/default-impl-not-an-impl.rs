//@ check-fail

#![feature(specialization)]
#![allow(incomplete_features)]

// Tests that a `default impl` does not count as an *actual* impl, so it cannot
// be used to satisfy trait bounds.

// A `default impl` may omit trait items, but a real impl may not.

trait Gapped {
    fn a(&self) -> u32;
    fn b(&self) -> u32;
}

default impl<T> Gapped for T {
    fn a(&self) -> u32 {
        1
    }
}

impl Gapped for u8 {}
//~^ ERROR not all trait items implemented, missing: `b`

// A `default impl` that defines *every* trait item is still not an impl.

trait Foo {
    fn f(&self) -> u32;
}

default impl<T> Foo for T {
    fn f(&self) -> u32 {
        1
    }
}

fn need_foo<T: Foo>(t: &T) -> u32 {
    t.f()
}

trait Bar {
    fn b(&self) -> u32;
}

impl<T: Foo> Bar for T {
    fn b(&self) -> u32 {
        self.f()
    }
}

fn need_bar<T: Bar>(t: &T) -> u32 {
    t.b()
}

fn main() {
    // as a bound (UFCS `<u32 as Foo>::f` is the same trait-selection path, omitted)
    need_foo(&0u32);
    //~^ ERROR the trait bound `u32: Foo` is not satisfied

    // as a method-probe candidate
    0u32.f();
    //~^ ERROR no method named `f` found for type `u32` in the current scope

    // when building a vtable
    let _: &dyn Foo = &0u32;
    //~^ ERROR the trait bound `u32: Foo` is not satisfied

    // transitively, as another impl's where-clause
    need_bar(&0i64);
    //~^ ERROR the trait bound `i64: Bar` is not satisfied
}
