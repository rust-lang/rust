//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no -Z verbose-internals

#![feature(rustc_attrs)]
#![rustc_dump_hidden_type_of_opaques]

trait Foo {
    fn hello(&self) -> impl Sized;
}

fn hello<'s, T: Foo>(x: &'s T) -> impl Sized + use<'s, T> {
    x.hello()
}
