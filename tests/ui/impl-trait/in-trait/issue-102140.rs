// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

trait Marker {}
impl Marker for u32 {}

trait MyTrait {
    fn foo(&self) -> impl Marker
    where
        Self: Sized;
}

struct Outer;

impl MyTrait for Outer {
    fn foo(&self) -> impl Marker {
        42
    }
}

impl dyn MyTrait {
    fn other(&self) -> impl Marker {
        MyTrait::foo(&self)
        //~^ ERROR the trait bound `&dyn MyTrait: MyTrait` is not satisfied
        //~| ERROR the trait bound `&dyn MyTrait: MyTrait` is not satisfied
        //~| ERROR the trait bound `&dyn MyTrait: MyTrait` is not satisfied
    }
}

fn main() {}
