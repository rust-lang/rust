#![feature(associated_type_bounds)]

trait A {
    type T;
}

trait B: A<T: B> {}
//~^ ERROR cycle detected when computing the implied predicates of `B`

fn main() {}
