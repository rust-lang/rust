//@ check-pass
//@ edition:2018

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    async fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
}

type Alias = Foo<'static>;
impl Alias {
    async fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
}

fn main() {}
