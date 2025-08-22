#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

trait Baz {}

impl Baz for u32 {}

type Qux = impl Baz;

trait Foo {
    type Assoc;
}

impl Foo for u32 {
    type Assoc = Qux;
}

#[define_opaque(Qux)]
fn assign() -> Qux {
    1
}

extern "C" {
    fn lint_me() -> <u32 as Foo>::Assoc; //~ ERROR: uses type `Qux`
}

fn main() {}
