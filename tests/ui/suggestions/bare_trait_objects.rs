//@ revisions: rust2015 rust2021
//@[rust2015] edition:2015
//@[rust2021] edition:2021


trait Foo {
    type Clone;
    fn foo() -> Clone;
    //[rust2021]~^ ERROR expected a type, found a trait
    //[rust2021]~| HELP `Clone` is dyn-incompatible, use `impl Clone` to return an opaque type, as long as you return a single underlying type
    //[rust2015]~^^^ WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //[rust2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //[rust2015]~| HELP if this is a dyn-compatible trait, use `dyn`
    //[rust2015]~| WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //[rust2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //[rust2015]~| HELP if this is a dyn-compatible trait, use `dyn`
    //[rust2015]~| ERROR the trait `Clone` is not dyn compatible [E0038]
    //~| HELP there is an associated type with the same name
    //[rust2015]~| HELP use `Self` to refer to the implementing type
}

trait DbHandle: Sized {}

trait DbInterface {
    type DbHandle;
    fn handle() -> DbHandle;
    //[rust2021]~^ ERROR expected a type, found a trait
    //[rust2021]~| HELP `DbHandle` is dyn-incompatible, use `impl DbHandle` to return an opaque type, as long as you return a single underlying type
    //[rust2015]~^^^ WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //[rust2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //[rust2015]~| HELP if this is a dyn-compatible trait, use `dyn`
    //[rust2015]~| WARNING trait objects without an explicit `dyn` are deprecated [bare_trait_objects]
    //[rust2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
    //[rust2015]~| HELP if this is a dyn-compatible trait, use `dyn`
    //[rust2015]~| ERROR the trait `DbHandle` is not dyn compatible [E0038]
    //~| HELP there is an associated type with the same name
    //[rust2015]~| HELP use `Self` to refer to the implementing type
}

fn main() {}
