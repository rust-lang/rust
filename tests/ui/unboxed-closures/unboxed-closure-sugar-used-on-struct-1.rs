// Test that parentheses form doesn't work with struct types appearing in local variables.

struct Bar<A> {
    f: A
}

fn bar() {
    let x: Box<Bar()> = panic!();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| ERROR struct takes 1 generic argument but 0 generic arguments
}

fn main() { }
