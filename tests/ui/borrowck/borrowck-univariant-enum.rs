// run-pass
#![allow(non_camel_case_types)]

use std::cell::Cell;

#[derive(Copy, Clone)]
enum newtype {
    newvar(isize)
}

pub fn main() {

    // Test that borrowck treats enums with a single variant
    // specially.

    let x = &Cell::new(5);
    let y = &Cell::new(newtype::newvar(3));
    let z = match y.get() {
      newtype::newvar(b) => {
        x.set(x.get() + 1);
        x.get() * b
      }
    };
    assert_eq!(z, 18);
}
