// FIXME: Investigate why `self_lifetime.rs` is check-pass but this isn't.

// edition:2018

#![feature(async_await)]

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    async fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
    //~^ ERROR missing lifetime specifier
    //~| ERROR cannot infer an appropriate lifetime
}

type Alias = Foo<'static>;
impl Alias {
    async fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
    //~^ ERROR lifetime mismatch
}

fn main() {}
