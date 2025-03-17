//@ check-pass

struct Foo<'a>(&'a str);

impl<'b> Foo<'b> {
    fn a<'a>(self: Self, a: &'a str) -> &str {
        //~^ WARNING lifetime flowing from input to output with different syntax
        a
    }
    fn b<'a>(self: Foo<'b>, a: &'a str) -> &str {
        //~^ WARNING lifetime flowing from input to output with different syntax
        a
    }
}

struct Foo2<'a>(&'a u32);
impl<'a> Foo2<'a> {
    fn foo(self: &Self) -> &u32 { self.0 } // ok
    fn bar(self: &Foo2<'a>) -> &u32 { self.0 } // ok (do not look into `Foo`)
    fn baz2(self: Self, arg: &u32) -> &u32 { arg } // use lt from `arg`
    fn baz3(self: Foo2<'a>, arg: &u32) -> &u32 { arg } // use lt from `arg`
}

fn main() {}
