//@ revisions: rpass1 cfail2

#![feature(type_alias_impl_trait)]

pub type Foo = impl Sized;

#[cfg_attr(rpass1, define_opaque())]
#[cfg_attr(cfail2, define_opaque(Foo))]
fn a() {
    //[cfail2]~^ ERROR item does not constrain `Foo::{opaque#0}`
    let _: Foo = b();
}

#[define_opaque(Foo)]
fn b() -> Foo {
    ()
}

fn main() {}
