//@ check-pass
//@ edition:2018

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    async fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
    //~^ WARNING lifetime flowing from input to output with different syntax
}

type Alias = Foo<'static>;
impl Alias {
    async fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
    //~^ WARNING lifetime flowing from input to output with different syntax
}

fn main() {}
