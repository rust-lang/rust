trait CollectionFamily {
    type Member<T>;
}
fn floatify() {
    Box::new(Family) as &dyn CollectionFamily<Member=usize>
    //~^ ERROR: missing generics for associated type
    //~| ERROR: the trait `CollectionFamily` is not dyn compatible
}

struct Family;

fn main() {}
