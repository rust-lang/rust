#![feature(type_alias_impl_trait)]

fn main() {}

trait T {
    type Assoc;
}

type Foo = impl T;

fn a() -> Foo {
    //~^ ERROR trait `T` is not implemented for `()`
    // This is not a defining use, it doesn't actually constrain the opaque type.
    panic!()
}
