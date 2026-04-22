//@ revisions: rpass1 bfail2

#![feature(type_alias_impl_trait)]

pub type Foo = impl Sized;

#[cfg_attr(rpass1, define_opaque())]
#[cfg_attr(bfail2, define_opaque(Foo))]
fn a() {
    //[bfail2]~^ ERROR item does not constrain `Foo::{opaque#0}`
    let _: Foo = b();
}

#[define_opaque(Foo)]
fn b() -> Foo {
    ()
}

fn main() {}
