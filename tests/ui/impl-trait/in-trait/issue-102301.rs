//@ check-pass

trait Foo<T> {
    fn foo<F2: Foo<T>>(self) -> impl Foo<T>;
}

struct Bar;

impl Foo<u8> for Bar {
    fn foo<F2: Foo<u8>>(self) -> impl Foo<u8> {
        self
    }
}

fn main() {}
