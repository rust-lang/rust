//@ proc-macro: test-macros.rs
//@ needs-unwind proc macro panics to report errors

#[macro_use]
extern crate test_macros;

#[derive(Identity, Panic)] //~ ERROR proc-macro derive panicked
struct Baz {
    //~^ ERROR the name `Baz` is defined multiple times
    a: i32,
    b: i32,
}

fn main() {}
