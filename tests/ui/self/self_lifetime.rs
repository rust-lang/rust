//@ check-pass

// https://github.com/rust-lang/rust/pull/60944#issuecomment-495346120

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
    //~^ WARNING elided lifetime has a name
}

type Alias = Foo<'static>;
impl Alias {
    fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
    //~^ WARNING elided lifetime has a name
}

fn main() {}
