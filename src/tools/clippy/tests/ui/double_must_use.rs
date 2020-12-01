#![warn(clippy::double_must_use)]
#![allow(clippy::result_unit_err)]

#[must_use]
pub fn must_use_result() -> Result<(), ()> {
    unimplemented!();
}

#[must_use]
pub fn must_use_tuple() -> (Result<(), ()>, u8) {
    unimplemented!();
}

#[must_use]
pub fn must_use_array() -> [Result<(), ()>; 1] {
    unimplemented!();
}

#[must_use = "With note"]
pub fn must_use_with_note() -> Result<(), ()> {
    unimplemented!();
}

fn main() {
    must_use_result();
    must_use_tuple();
    must_use_with_note();
}
