//@ edition:2018
//@ aux-crate:reexported_trait=reexported-trait.rs

fn main() {
    reexported_trait::FooStruct.trait_method();
    //~^ ERROR
    reexported_trait::FooStruct.trait_method_b();
    //~^ ERROR
}
