//@ edition:2021

trait Foo {
    type Clone; //~ NOTE you might have meant to use this associated type
    fn foo() -> Clone;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| HELP `Clone` is not object safe, use `impl Clone` to return an opaque type, as long as you return a single underlying type
    //~| HELP there is an associated type with the same name
}

trait DbHandle: Sized {}

trait DbInterface {
    type DbHandle; //~ NOTE you might have meant to use this associated type
    fn handle() -> DbHandle;
    //~^ ERROR trait objects must include the `dyn` keyword
    //~| HELP `DbHandle` is not object safe, use `impl DbHandle` to return an opaque type, as long as you return a single underlying type
    //~| HELP there is an associated type with the same name
}

fn main() {}
