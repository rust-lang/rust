//! Ensure that adding rustc_isolated_const makes calls
//! to local methods fail.

#![feature(const_trait_impl, rustc_attrs)]

#[rustc_isolated_const]
const VAL: usize = {
    <() as Foo>::bar()
    //~^ ERROR the trait bound `(): Foo` is not satisfied
};

#[rustc_isolated_const]
const VAL2: () = {
    Bar.bar()
    //^ FIXME(isolated_const) should also error
};

const trait Foo {
    fn bar() -> usize {
        todo!()
    }
}

const impl Foo for () {}

struct Bar;

const impl Bar {
    fn bar(&self) {}
}

fn main() {
    assert_eq!(VAL, 42);
}
