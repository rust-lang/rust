// Test that parentheses form doesn't work with struct types appearing in local variables.

struct Bar<A> {
    f: A
}

fn bar() {
    let x: Box<Bar()> = panic!();
    //~^ ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| ERROR wrong number of type arguments: expected 1, found 0
}

fn main() { }
