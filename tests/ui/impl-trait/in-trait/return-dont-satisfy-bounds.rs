trait Foo<T> {
    fn foo<F2>(self) -> impl Foo<T>;
}

struct Bar;

impl Foo<char> for Bar {
    fn foo<F2: Foo<u8>>(self) -> impl Foo<u8> {
        //~^ ERROR: the trait bound `impl Foo<u8>: Foo<char>` is not satisfied [E0277]
        //~| ERROR: the trait bound `Bar: Foo<u8>` is not satisfied [E0277]
        //~| ERROR: impl has stricter requirements than trait
        self
    }
}

fn main() {}
