trait Foo {
    fn bar() -> impl std::fmt::Display;
}

impl Foo for () {
    fn bar() -> () {}
    //~^ ERROR `()` doesn't implement `std::fmt::Display`
}

fn main() {}
