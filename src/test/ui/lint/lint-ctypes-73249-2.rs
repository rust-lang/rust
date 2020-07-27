#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

pub trait Baz { }

impl Baz for () { }

type Qux = impl Baz;

fn assign() -> Qux {}

pub trait Foo {
    type Assoc: 'static;
}

impl Foo for () {
    type Assoc = Qux;
}

#[repr(transparent)]
pub struct A<T: Foo> {
    x: &'static <T as Foo>::Assoc,
}

extern "C" {
    pub fn lint_me() -> A<()>; //~ ERROR: uses type `impl Baz`
}

fn main() {}
