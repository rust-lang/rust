use std::fmt::Debug;

fn testing<T: Debug>(t: T) {}

struct Foo;

fn main() {
    testing((1, Foo));
    //~^ ERROR `Foo` doesn't implement `Debug` [E0277]
}
