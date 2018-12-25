trait Foo {
    fn foo<T: Default>(x: T) -> Self;
}

struct Bar;

impl Foo for Bar {
    fn foo(x: bool) -> Self { Bar } //~ ERROR E0049
}

fn main() {
}
