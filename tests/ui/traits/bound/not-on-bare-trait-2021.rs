//@ edition:2021
trait Foo {
    fn dummy(&self) {}
}

// This should emit the less confusing error, not the more confusing one.

fn foo(_x: Foo + Send) {
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR size for values of type
}
fn bar(x: Foo) -> Foo {
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    //~| ERROR size for values of type
    x
}

fn main() {}
