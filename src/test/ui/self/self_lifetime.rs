// compile-pass

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
}

type Alias = Foo<'static>;
impl Alias {
    fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
}

fn main() {}
