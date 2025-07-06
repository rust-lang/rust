//! This test checks two common compilation errors related to generic enums

enum Quux<T> {
    //~^ ERROR: parameter `T` is never used
    Bar,
}

fn foo(c: Quux) {
    //~^ ERROR missing generics for enum `Quux`
    assert!((false));
}

fn main() {
    panic!();
}
