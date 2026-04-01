// FIXME(compiler-errors): I'm not exactly sure if this is expected to pass or not.
// But we fixed an ICE anyways.

#![feature(specialization)]
#![allow(incomplete_features)]

trait Foo {
    fn bar(&self) -> impl Sized;
}

default impl<U> Foo for U
where
    U: Copy,
{
    fn bar(&self) -> U {
        //~^ ERROR method `bar` has an incompatible type for trait
        //~| ERROR method with return-position `impl Trait` in trait cannot be specialized
        *self
    }
}

impl Foo for i32 {}

fn main() {
    1i32.bar();
}
