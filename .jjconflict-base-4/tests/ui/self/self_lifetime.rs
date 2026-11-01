//@ check-pass

// https://github.com/rust-lang/rust/pull/60944#issuecomment-495346120

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
    //~^ WARNING eliding a lifetime that's named elsewhere is confusing
}

type Alias = Foo<'static>;
impl Alias {
    fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
    //~^ WARNING eliding a lifetime that's named elsewhere is confusing
}

fn main() {}
