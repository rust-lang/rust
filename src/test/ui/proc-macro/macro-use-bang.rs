// compile-pass
// aux-build:test-macros.rs

#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate test_macros;

fn main() {
    identity!(println!("Hello, world!"));
}
