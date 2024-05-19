trait Foo {}

impl Foo for i32 {}

fn needs_foo(_: impl Foo) {}

fn test(x: &Box<dyn Fn() -> i32>) {
    needs_foo(x);
    //~^ ERROR the trait bound
    //~| HELP use parentheses to call this trait object
}

fn main() {}
