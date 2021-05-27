#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait CollectionFamily {
    type Member<T>;
}
fn floatify() {
    Box::new(Family) as &dyn CollectionFamily<Member=usize>
    //~^ ERROR: missing generics for associated type
    //~| ERROR: the trait `CollectionFamily` cannot be made into an object
}

struct Family;

fn main() {}
