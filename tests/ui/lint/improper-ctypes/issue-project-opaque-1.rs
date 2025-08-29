#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

// Issue: https://github.com/rust-lang/rust/issues/73251
// Decisions on whether projections that normalize to opaque types then to something else
// should warn or not

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
