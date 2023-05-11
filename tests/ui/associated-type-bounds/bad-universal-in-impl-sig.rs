#![feature(associated_type_bounds)]

trait Trait {
    type Item;
}

trait Trait2 {}

// It's not possible to insert a universal `impl Trait` here!
impl dyn Trait<Item: Trait2> {}
//~^ ERROR associated type bounds are only allowed in where clauses and function signatures

fn main() {}
