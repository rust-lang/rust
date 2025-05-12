trait Bar {}

impl Bar for i32 {}

struct Qux;

impl Bar for Qux {}

fn foo() -> impl Bar {
    //~^ ERROR the trait bound `(): Bar` is not satisfied
    5;
    //~^ HELP remove this semicolon
}

fn bar() -> impl Bar {
    //~^ ERROR the trait bound `(): Bar` is not satisfied
    //~| HELP the following other types implement trait `Bar`:
    "";
}

fn main() {}
