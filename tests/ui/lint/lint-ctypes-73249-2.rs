#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

trait Baz {}

impl Baz for () {}

type Qux = impl Baz;

fn assign() -> Qux {}

trait Foo {
    type Assoc: 'static;
}

impl Foo for () {
    type Assoc = Qux;
}

#[repr(transparent)]
struct A<T: Foo> {
    x: &'static <T as Foo>::Assoc,
}

extern "C" {
    fn lint_me() -> A<()>; //~ ERROR: uses type `Qux`
}

fn main() {}
