trait Foo {}

trait Bar<T> {}

trait Iterable {
    type Item;
}

struct Container<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` only allowed in function and inherent method return types
    field: T
}

enum Enum<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` only allowed in function and inherent method return types
    A(T),
}

union Union<T: Iterable<Item = impl Foo> + Copy> {
    //~^ ERROR `impl Trait` only allowed in function and inherent method return types
    x: T,
}

type Type<T: Iterable<Item = impl Foo>> = T;
//~^ ERROR `impl Trait` only allowed in function and inherent method return types

fn main() {
}
