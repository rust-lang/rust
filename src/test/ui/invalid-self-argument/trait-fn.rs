struct Foo {}

impl Foo {
    fn c(foo: u32, self) {}
    //~^ ERROR unexpected `self` argument in function
    //~| NOTE `self` is only valid as the first argument of an associated function

    fn good(&mut self, foo: u32) {}
}

fn main() { }
