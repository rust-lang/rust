//@ edition:2021
trait Foo {
    fn dummy(&self) {}
}

// This should emit the less confusing error, not the more confusing one.

fn foo(_x: Foo + Send) {
    //~^ ERROR size for values of type
}
fn bar(x: Foo) -> Foo {
    //~^ ERROR size for values of type
    x
}
fn bat(x: &Foo) -> Foo {
    //~^ ERROR return type cannot have an unboxed trait object
    //~| ERROR trait objects must include the `dyn` keyword
    x
}
fn bae(x: &Foo) -> &Foo {
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| ERROR trait objects must include the `dyn` keyword
    x
}
fn qux() -> Foo {
    //~^ ERROR return type cannot have an unboxed trait object
    todo!()
}

fn main() {}
