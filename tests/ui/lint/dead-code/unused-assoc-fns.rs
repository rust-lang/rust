#![deny(unused)]

struct Foo;

impl Foo {
    fn one() {}
    //~^ ERROR associated function `one` is never used [dead_code]

    fn two(&self) {}
    //~^ ERROR method `two` is never used [dead_code]

    // seperation between functions
    // ...
    // ...

    fn used() {}

    fn three(&self) {
    //~^ ERROR method `three` is never used [dead_code]
        Foo::one();
        // ...
    }
}

fn main() {
    Foo::used();
}
