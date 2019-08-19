// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive(Identity, Panic)] //~ ERROR proc-macro derive panicked
struct Baz {
    a: i32,
    b: i32,
}

fn main() {}
