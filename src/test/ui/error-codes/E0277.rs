// ignore-cloudabi no std::path

use std::path::Path;

trait Foo {
    fn bar(&self);
}

fn some_func<T: Foo>(foo: T) {
    foo.bar();
}

fn f(p: Path) { }
//~^ ERROR the size for values of type

fn main() {
    some_func(5i32);
    //~^ ERROR the trait bound `i32: Foo` is not satisfied
}
