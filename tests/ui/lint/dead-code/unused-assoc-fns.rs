#![deny(unused)]

struct Foo;

impl Foo {
    fn one() {}
    //~^ ERROR associated functions `one`, `two`, and `three` are never used [dead_code]

    fn two(&self) {}

    // seperation between functions
    // ...
    // ...

    fn used() {}

    fn three(&self) {
        Foo::one();
        // ...
    }
}

fn main() {
    Foo::used();
}
