// check-pass
// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

fn main() {
    identity!(println!("Hello, world!"));
}
