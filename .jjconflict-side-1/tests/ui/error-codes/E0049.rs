trait Foo {
    fn foo<T: Default>(x: T) -> Self;
}

struct Bar;

impl Foo for Bar {
    fn foo(x: bool) -> Self { Bar } //~ ERROR E0049
}

trait Fuzz {
    fn fuzz<A: Default, B>(x: A, y: B) -> Self;
}

struct Baz;

impl Fuzz for Baz {
    fn fuzz(x: bool, y: bool) -> Self { Baz } //~ ERROR E0049
}

fn main() {
}
