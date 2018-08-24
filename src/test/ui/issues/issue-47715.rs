trait Foo {}

trait Bar<T> {}

trait Iterable {
    type Item;
}

struct Container<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` not allowed
    field: T
}

enum Enum<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` not allowed
    A(T),
}

union Union<T: Iterable<Item = impl Foo> + Copy> {
    //~^ ERROR `impl Trait` not allowed
    x: T,
}

type Type<T: Iterable<Item = impl Foo>> = T;
//~^ ERROR `impl Trait` not allowed

fn main() {
}
