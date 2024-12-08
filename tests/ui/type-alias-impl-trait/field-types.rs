#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

//@ check-pass

use std::fmt::Debug;

type Foo = impl Debug;

struct Bar {
    foo: Foo,
}

fn bar() -> Bar {
    Bar { foo: "foo" }
}

fn main() {}
