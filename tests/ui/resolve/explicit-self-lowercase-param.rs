struct Foo;

impl Foo {
    fn do_nothing(self: Box<self>) {} //~ ERROR attempt to use a non-constant value in a constant
    //~^ HELP try using `Self`
}

fn main() {}
