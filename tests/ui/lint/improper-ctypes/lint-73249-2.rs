//@ check-pass  // possible FIXME: see below

#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

trait Baz {}

impl Baz for () {}

type Qux = impl Baz;

#[define_opaque(Qux)]
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
    // possible FIXME(ctypes): the unsafety of Qux is unseen, as it is behing a FFI-safe indirection
    fn lint_me() -> A<()>;
}

fn main() {}
