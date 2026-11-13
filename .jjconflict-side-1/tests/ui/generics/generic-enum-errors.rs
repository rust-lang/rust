//! This test checks that unused generics are rejected by compiler

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
