//@ check-pass

// https://github.com/rust-lang/rust/pull/60944#issuecomment-495346120

struct Foo<'a>(&'a ());
impl<'a> Foo<'a> {
    fn foo<'b>(self: &'b Foo<'a>) -> &() { self.0 }
    //~^ WARNING lifetime flowing from input to output with different syntax
}

type Alias = Foo<'static>;
impl Alias {
    fn bar<'a>(self: &Alias, arg: &'a ()) -> &() { arg }
    //~^ WARNING lifetime flowing from input to output with different syntax
}

fn main() {}
