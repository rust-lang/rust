//@ aux-build:enums.rs

extern crate enums;

use enums::FieldLessWithNonExhaustiveVariant;

fn main() {
    let e = FieldLessWithNonExhaustiveVariant::default();
    let d = e as u8; //~ ERROR casting `FieldLessWithNonExhaustiveVariant` as `u8` is invalid [E0606]
    assert_eq!(d, 0);
}
