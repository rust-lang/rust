trait Foo {
    const ASSOC: bool = true;
}
impl<T> Foo for fn(T) {}

fn foo(_x: i32) {}

fn impls_foo<T: Foo>(_x: T) {}

fn main() {
    impls_foo(foo as fn(i32));

    <fn(&u8) as Foo>::ASSOC;
    //~^ ERROR implementation of `Foo` is not general enough
}
