//@ edition: 2015

trait Foo {
    type Clone;
    fn foo() -> Clone;
    //~^ WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| HELP if this is a dyn-compatible trait, use `dyn`
    //~| WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| HELP if this is a dyn-compatible trait, use `dyn`
    //~| ERROR the trait `Clone` is not dyn compatible [E0038]
    //~| HELP there is an associated type with the same name
}

trait DbHandle: Sized {}

trait DbInterface {
    type DbHandle;
    fn handle() -> DbHandle;
    //~^ WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| HELP if this is a dyn-compatible trait, use `dyn`
    //~| WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| HELP if this is a dyn-compatible trait, use `dyn`
    //~| ERROR the trait `DbHandle` is not dyn compatible [E0038]
    //~| HELP there is an associated type with the same name
}

fn main() {}
