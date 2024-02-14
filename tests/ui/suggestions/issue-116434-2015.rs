trait Foo {
    type Clone;
    fn foo() -> Clone;
    //~^ WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| ERROR the trait `Clone` cannot be made into an object [E0038]
}

trait DbHandle: Sized {}

trait DbInterface {
    type DbHandle;
    fn handle() -> DbHandle;
    //~^ WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //~| ERROR the trait `DbHandle` cannot be made into an object [E0038]
}

fn main() {}
