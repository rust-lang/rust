// run-pass

#![feature(const_fn, core_intrinsics)]

use std::{intrinsics::caller_location, panic::Location};

const LOCATION: &Location = caller_location();
const NESTED: &Location = {
    const fn nested_location() -> &'static Location<'static> {
        caller_location()
    };
    nested_location()
};

fn main() {
    assert_eq!(LOCATION.file(), file!());
    assert_eq!(LOCATION.line(), 7);
    assert_eq!(LOCATION.column(), 29);

    assert_eq!(NESTED.file(), file!());
    assert_eq!(NESTED.line(), 10);
    assert_eq!(NESTED.column(), 9);
}
