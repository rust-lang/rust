trait Bar {}

impl Bar for i32 {}

struct Qux;

impl Bar for Qux {}

fn foo() -> impl Bar {
    //~^ ERROR trait `Bar` is not implemented for `()`
    5;
    //~^ HELP remove this semicolon
}

fn bar() -> impl Bar {
    //~^ ERROR trait `Bar` is not implemented for `()`
    //~| HELP the following other types implement trait `Bar`:
    "";
}

fn main() {}
