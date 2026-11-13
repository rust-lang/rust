trait TraitB {
    type Item;
}

trait TraitA<A> {
    fn foo<T: TraitB<Item = A>>(_: T) -> Self;
    fn bar<T>(_: T) -> Self;
    fn baz<T>(_: T) -> Self where T: TraitB, <T as TraitB>::Item: Copy;
    fn bat<T: TraitB<Item: Copy>>(_: T) -> Self;
}

struct S;

impl TraitA<()> for S { //~ ERROR not all trait items implemented
}

use std::iter::FromIterator;
struct X;
impl FromIterator<()> for X { //~ ERROR not all trait items implemented
}

fn main() {}
