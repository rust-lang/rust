trait Foo {}

trait Bar<T> {}

trait Iterable {
    type Item;
}

struct Container<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` isn't allowed within generic [E0562]
    field: T
}

enum Enum<T: Iterable<Item = impl Foo>> {
    //~^ ERROR `impl Trait` isn't allowed within generic [E0562]
    A(T),
}

union Union<T: Iterable<Item = impl Foo> + Copy> {
    //~^ ERROR `impl Trait` isn't allowed within generic [E0562]
    x: T,
}

type Type<T: Iterable<Item = impl Foo>> = T;
//~^ ERROR `impl Trait` isn't allowed within generic [E0562]

fn main() {
}
