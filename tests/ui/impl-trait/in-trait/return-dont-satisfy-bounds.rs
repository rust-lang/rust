trait Foo<T> {
    fn foo<F2>(self) -> impl Foo<T>;
}

struct Bar;

impl Foo<char> for Bar {
    fn foo<F2: Foo<u8>>(self) -> impl Foo<u8> {
        //~^ ERROR trait `Foo<char>` is not implemented for `impl Foo<u8>`
        //~| ERROR trait `Foo<u8>` is not implemented for `Bar`
        //~| ERROR: impl has stricter requirements than trait
        self
    }
}

fn main() {}
