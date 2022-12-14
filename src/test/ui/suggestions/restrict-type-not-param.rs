use std::ops::Add;

struct Wrapper<T>(T);

trait Foo {}

fn qux<T>(a: Wrapper<T>, b: T) -> T {
    a + b
    //~^ ERROR cannot add `T` to `Wrapper<T>`
}

fn main() {}
