// run-pass
// revisions: default mir-opt
//[mir-opt] compile-flags: -Zmir-opt-level=3

#![feature(const_caller_location, const_fn)]

use std::panic::Location;

const LOCATION: &Location = Location::caller();

const TRACKED: &Location = tracked();
#[track_caller]
const fn tracked() -> &'static Location <'static> {
    Location::caller()
}

const NESTED: &Location = nested_location();
const fn nested_location() -> &'static Location<'static> {
    Location::caller()
}

const CONTAINED: &Location = contained();
const fn contained() -> &'static Location<'static> {
    tracked()
}

fn main() {
    assert_eq!(LOCATION.file(), file!());
    assert_eq!(LOCATION.line(), 9);
    assert_eq!(LOCATION.column(), 29);

    assert_eq!(TRACKED.file(), file!());
    assert_eq!(TRACKED.line(), 11);
    assert_eq!(TRACKED.column(), 28);

    assert_eq!(NESTED.file(), file!());
    assert_eq!(NESTED.line(), 19);
    assert_eq!(NESTED.column(), 5);

    assert_eq!(CONTAINED.file(), file!());
    assert_eq!(CONTAINED.line(), 24);
    assert_eq!(CONTAINED.column(), 5);
}
