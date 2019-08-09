// build-pass (FIXME(62277): could be check-pass?)
// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr]
struct Foo;

fn main() {
    let _ = Foo;
}
