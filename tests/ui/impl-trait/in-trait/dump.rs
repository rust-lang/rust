//@ compile-flags: -Zverbose-internals

#![feature(precise_capturing_in_traits, rustc_attrs)]
#![rustc_hidden_type_of_opaques]

trait Foo {
    fn hello(&self) -> impl Sized;
}

fn hello<'s, T: Foo>(x: &'s T) -> impl Sized + use<'s, T> {
//~^ ERROR <T as Foo>::{synthetic#0}<'s/#1>
    x.hello()
}

fn main() {}
