//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr]
struct Foo;

fn main() {
    let _ = Foo;
}
