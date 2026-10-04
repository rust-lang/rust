//@ edition:2021

trait Foo {
    type Clone;
    fn foo() -> Clone;
    //~^ ERROR expected a type, found a trait
    //~| HELP `Clone` is dyn-incompatible, use `impl Clone` to return an opaque type, as long as you return a single underlying type
    //~| HELP there is an associated type with the same name
}

trait DbHandle: Sized {}

trait DbInterface {
    type DbHandle;
    fn handle() -> DbHandle;
    //~^ ERROR expected a type, found a trait
    //~| HELP `DbHandle` is dyn-incompatible, use `impl DbHandle` to return an opaque type, as long as you return a single underlying type
    //~| HELP there is an associated type with the same name
}

fn main() {}
