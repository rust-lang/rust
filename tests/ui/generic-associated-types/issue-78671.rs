// revisions: base extended

#![cfg_attr(extended, feature(generic_associated_types_extended))]
#![cfg_attr(extended, allow(incomplete_features))]

trait CollectionFamily {
    type Member<T>;
}
fn floatify() {
    Box::new(Family) as &dyn CollectionFamily<Member=usize>
    //~^ ERROR: missing generics for associated type
    //[base]~^^ ERROR: the trait `CollectionFamily` cannot be made into an object
}

struct Family;

fn main() {}
