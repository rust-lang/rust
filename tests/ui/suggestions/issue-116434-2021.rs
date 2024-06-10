//@ edition:2021

trait Foo {
    type Clone;
    fn foo() -> Clone;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| HELP `Clone` is not object safe, use `impl Clone` to return an opaque type, as long as you return a single underlying type
}

trait DbHandle: Sized {}

trait DbInterface {
    type DbHandle;
    fn handle() -> DbHandle;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| HELP `DbHandle` is not object safe, use `impl DbHandle` to return an opaque type, as long as you return a single underlying type
}

fn main() {}
