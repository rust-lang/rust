//@ compile-flags: -Zverbose-internals

#![feature(rustc_attrs)]
#![rustc_hidden_type_of_opaques]

trait Foo {
    fn hello(&self) -> impl Sized;
}

fn hello<'s, T: Foo>(x: &'s T) -> impl Sized + use<'s, T> {
    //~^ ERROR <T as Foo>::hello::{anon_assoc#0}<'s/#1>
    x.hello()
}

fn main() {}
