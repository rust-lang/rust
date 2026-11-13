//@ check-pass

#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

pub(crate) type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
pub(crate) fn foo() -> Foo {
    22_u32
}

fn is_send<T: Send>(_: T) {}

fn main() {
    is_send(foo());
}
