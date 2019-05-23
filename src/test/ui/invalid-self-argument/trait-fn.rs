struct Foo {}

impl Foo {
    fn c(foo: u32, self) {}
    //~^ ERROR unexpected `self` argument in function
    //~| NOTE must be the first associated function argument

    fn good(&mut self, foo: u32) {}
}

fn main() { }
