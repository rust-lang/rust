// https://github.com/rust-lang/rust/issues/56175
//@ edition:2018
//@ aux-crate:reexported_trait=reexported-trait-56175.rs

fn main() {
    reexported_trait::FooStruct.trait_method();
    //~^ ERROR
    reexported_trait::FooStruct.trait_method_b();
    //~^ ERROR
}
