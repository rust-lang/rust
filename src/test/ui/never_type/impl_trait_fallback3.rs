#![feature(type_alias_impl_trait)]

fn main() {}

trait T {
    type Assoc;
}

type Foo = impl T;
//~^ ERROR could not find defining uses

fn a() -> Foo {
    // This is not a defining use, it doesn't actually constrain the opaque type.
    panic!()
}
