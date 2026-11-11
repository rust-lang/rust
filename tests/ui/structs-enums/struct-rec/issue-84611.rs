struct Foo<T> {
//~^ ERROR recursive type `Foo` has infinite size
    x: Foo<[T; 1]>,
    y: T,
}

struct Bar {
    x: Foo<Bar>,
}

fn main() {}
