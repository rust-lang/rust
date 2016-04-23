#![feature(custom_attribute, specialization)]
#![allow(dead_code, unused_attributes)]

trait IsUnit {
    fn is_unit() -> bool;
}

impl<T> IsUnit for T {
    default fn is_unit() -> bool { false }
}

impl IsUnit for () {
    fn is_unit() -> bool { true }
}

#[miri_run]
fn specialization() -> (bool, bool) {
    (i32::is_unit(), <()>::is_unit())
}

#[miri_run]
fn main() {
    assert_eq!(specialization(), (false, true));
}
