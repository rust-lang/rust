//@ check-pass

#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

// Issue: https://github.com/rust-lang/rust/issues/73249
// Decisions on whether projections that normalize to opaque types then to something else
// should warn or not

trait Foo {
    type Assoc;
}

impl Foo for () {
    type Assoc = u32;
}

type Bar = impl Foo<Assoc = u32>;

#[define_opaque(Bar)]
fn assign() -> Bar {}

extern "C" {
    fn lint_me() -> <Bar as Foo>::Assoc;
}

fn main() {}
