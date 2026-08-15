#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

//@ check-pass

pub type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
pub fn foo() -> Foo {
    is_send(bar())
}

pub fn bar() {
    is_send(foo());
}

#[define_opaque(Foo)]
fn baz() -> Foo {
    ()
}

fn is_send<T: Send>(_: T) {}

fn main() {}
