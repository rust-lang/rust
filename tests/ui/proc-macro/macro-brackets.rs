// aux-build:test-macros.rs

#[macro_use]
extern crate test_macros;

macro_rules! id {
    ($($t:tt)*) => ($($t)*)
}

#[identity_attr]
id![static X: u32 = 'a';]; //~ ERROR: mismatched types


fn main() {}
