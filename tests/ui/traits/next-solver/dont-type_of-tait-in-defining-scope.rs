// revisions: is_send not_send
// compile-flags: -Znext-solver

#![feature(type_alias_impl_trait)]

#[cfg(is_send)]
type Foo = impl Send;

#[cfg(not_send)]
type Foo = impl Sized;

fn needs_send<T: Send>() {}

fn test(_: Foo) {
    needs_send::<Foo>();
    //~^ ERROR type annotations needed: cannot satisfy `Foo == _`
}

fn defines(_: Foo) {
    let _: Foo = ();
}

fn main() {}
