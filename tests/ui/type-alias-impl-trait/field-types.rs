//! Show that `defines(StructName)` works for
//! fields of that struct being an opaque type.

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

//@ check-pass

use std::fmt::Debug;

type Foo = impl Debug;

struct Bar {
    foo: Foo,
}

#[define_opaque(Bar)]
fn bar() -> Bar {
    Bar { foo: "foo" }
}

fn main() {}
