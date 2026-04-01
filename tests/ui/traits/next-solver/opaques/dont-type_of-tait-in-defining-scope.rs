//@ revisions: is_send not_send
//@ compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

#[cfg(is_send)]
type Foo = impl Send;

#[cfg(not_send)]
type Foo = impl Sized;

fn needs_send<T: Send>() {}

#[define_opaque(Foo)]
fn test(_: Foo) {
    //~^ ERROR type annotations needed
    needs_send::<Foo>();
}

#[define_opaque(Foo)]
fn defines() {
    let _: Foo = ();
}

fn main() {}
