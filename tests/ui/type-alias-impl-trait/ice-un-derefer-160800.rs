//! Regression test for #160800.

#![feature(type_alias_impl_trait)]

type Foo = impl Send;

#[define_opaque(Foo)]
const VALUE: Foo = todo!();
//~^ ERROR item does not constrain `Foo::{opaque#0}`

#[define_opaque(Foo)]
fn test(_foo: Foo) {
    match VALUE {
        &mut Some(ref mut x) => *x,
        &mut None => panic!(),
    }
}

fn main() {}
