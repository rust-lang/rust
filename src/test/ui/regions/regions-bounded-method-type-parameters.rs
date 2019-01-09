// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

struct Foo;

impl Foo {
    fn some_method<A:'static>(self) { }
}

fn caller<'a>(x: &isize) {
    Foo.some_method::<&'a isize>();
    //~^ ERROR does not fulfill the required lifetime
}

fn main() { }
