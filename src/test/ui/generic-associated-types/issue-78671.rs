#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait CollectionFamily {
    type Member<T>;
         //~^ ERROR: missing generics for associated type
}
fn floatify() {
    Box::new(Family) as &dyn CollectionFamily<Member=usize>
}

struct Family;

fn main() {}
