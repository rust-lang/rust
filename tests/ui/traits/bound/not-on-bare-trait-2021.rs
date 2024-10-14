//@ edition:2021
trait Foo {
    fn dummy(&self) {}
}

// This should emit the less confusing error, not the more confusing one.

fn foo(_x: Foo + Send) {
    //~^ ERROR expected a type, found a trait
}
fn bar(x: Foo) -> Foo {
    //~^ ERROR expected a type, found a trait
    //~| ERROR expected a type, found a trait
    x
}

fn main() {}
