#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

// FIXME This should compile, but it currently doesn't

use std::fmt::Debug;

type Foo = impl Debug;
//~^ ERROR: could not find defining uses

struct Bar {
    foo: Foo,
}

fn bar() -> Bar {
    Bar { foo: "foo" }
    //~^ ERROR: mismatched types [E0308]
}

fn main() {}
