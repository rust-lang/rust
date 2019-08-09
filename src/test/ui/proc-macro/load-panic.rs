// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[derive(Panic)]
//~^ ERROR: proc-macro derive panicked
struct Foo;

fn main() {}
