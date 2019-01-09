// aux-build:trait_safety_lib.rs

// Check that unsafe traits require unsafe impls and that inherent
// impls cannot be unsafe.

extern crate trait_safety_lib as lib;

struct Bar;
impl lib::Foo for Bar { //~ ERROR requires an `unsafe impl` declaration
    fn foo(&self) -> isize {
        panic!();
    }
}

fn main() { }
