// aux-build:derive-panic.rs
// compile-flags:--error-format human

#[macro_use]
extern crate derive_panic;

#[derive(A)]
//~^ ERROR: proc-macro derive panicked
struct Foo;

fn main() {}
