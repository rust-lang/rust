// run-pass

#![feature(const_fn, core_intrinsics, track_caller)]

use std::{intrinsics::caller_location, panic::Location};

const LOCATION: &Location = caller_location();

const TRACKED: &Location = tracked();
#[track_caller]
const fn tracked() -> &'static Location <'static> {
    caller_location()
}

const NESTED: &Location = nested_location();
const fn nested_location() -> &'static Location<'static> {
    caller_location()
}

const CONTAINED: &Location = contained();
const fn contained() -> &'static Location<'static> {
    tracked()
}

fn main() {
    assert_eq!(LOCATION.file(), file!());
    assert_eq!(LOCATION.line(), 7);
    assert_eq!(LOCATION.column(), 29);

    assert_eq!(TRACKED.file(), file!());
    assert_eq!(TRACKED.line(), 9);
    assert_eq!(TRACKED.column(), 28);

    assert_eq!(NESTED.file(), file!());
    assert_eq!(NESTED.line(), 17);
    assert_eq!(NESTED.column(), 5);

    assert_eq!(CONTAINED.file(), file!());
    assert_eq!(CONTAINED.line(), 22);
    assert_eq!(CONTAINED.column(), 5);
}
