//@ check-pass
//@ edition:2018

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    async fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
    //~^ WARNING eliding a lifetime that's named elsewhere is confusing
}

type Alias = Foo<'static>;
impl Alias {
    async fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
    //~^ WARNING eliding a lifetime that's named elsewhere is confusing
}

fn main() {}
