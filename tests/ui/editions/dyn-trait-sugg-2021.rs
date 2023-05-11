// edition:2021

trait Foo<T> {}

impl<T> dyn Foo<T> {
    fn hi(_x: T) {}
}

fn main() {
    Foo::hi(123);
    //~^ ERROR trait objects must include the `dyn` keyword
}
