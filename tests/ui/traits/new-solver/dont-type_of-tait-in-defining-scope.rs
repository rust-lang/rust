// revisions: is_send not_send
// compile-flags: -Ztrait-solver=next
//[is_send] check-pass

#![feature(type_alias_impl_trait)]

#[cfg(is_send)]
type Foo = impl Send;

#[cfg(not_send)]
type Foo = impl Sized;

fn needs_send<T: Send>() {}

fn test(_: Foo) {
    needs_send::<Foo>();
    //[not_send]~^ ERROR type annotations needed: cannot satisfy `Foo: Send`
}

fn defines(_: Foo) {
    let _: Foo = ();
}

fn main() {}
