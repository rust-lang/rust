// Test that parentheses form doesn't work with struct types appearing in argument types.

struct Bar<A> {
    f: A
}

fn foo(b: Box<Bar()>) {
    //~^ ERROR parenthesized parameters may only be used with a trait
    //~| ERROR the type placeholder `_` is not allowed within types on item signatures
}

fn main() { }
