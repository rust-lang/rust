//@ edition:2021

trait Foo<T> {}

impl<T> dyn Foo<T> {
    fn hi(_x: T) {}
}

fn main() {
    Foo::hi(123);
    //~^ ERROR expected a type, found a trait
}
