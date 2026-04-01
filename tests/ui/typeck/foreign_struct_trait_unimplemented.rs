//@ aux-build:foreign_struct_trait_unimplemented.rs

extern crate foreign_struct_trait_unimplemented;

pub trait Test {}

struct A;
impl Test for A {}

fn needs_test(_: impl Test) {}

fn main() {
    needs_test(foreign_struct_trait_unimplemented::B);
    //~^ ERROR the trait bound `B: Test` is not satisfied
}
