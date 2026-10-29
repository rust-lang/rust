//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

impl<R: Duh, F: FnMut() -> R> Trait for F {
    type Assoc = R;
}

type Sendable = impl Send + Duh;

type Foo = impl Trait<Assoc = Sendable>;

#[define_opaque(Foo)]
fn foo() -> Foo {
    || 42
}

fn main() {}
