struct Foo {}

impl Foo {
    fn new(_a: usize) -> Self { Foo{} }
}

static mut a: Foo = Foo::new(3);
//~^ ERROR cannot call non-const associated function

fn main() {}
