#![feature(track_caller, core_intrinsics)]

use std::panic::Location;

#[track_caller]
fn tracked() -> &'static Location<'static> {
    Location::caller() // most importantly, we never get line 7
}

fn nested_intrinsic() -> &'static Location<'static> {
    Location::caller()
}

fn nested_tracked() -> &'static Location<'static> {
    tracked()
}

macro_rules! caller_location_from_macro {
    () => (core::panic::Location::caller());
}

fn main() {
    let location = Location::caller();
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), 23);
    assert_eq!(location.column(), 20);

    let tracked = tracked();
    assert_eq!(tracked.file(), file!());
    assert_eq!(tracked.line(), 28);
    assert_eq!(tracked.column(), 19);

    let nested = nested_intrinsic();
    assert_eq!(nested.file(), file!());
    assert_eq!(nested.line(), 11);
    assert_eq!(nested.column(), 5);

    let contained = nested_tracked();
    assert_eq!(contained.file(), file!());
    assert_eq!(contained.line(), 15);
    assert_eq!(contained.column(), 5);

    // `Location::caller()` in a macro should behave similarly to `file!` and `line!`,
    // i.e. point to where the macro was invoked, instead of the macro itself.
    let inmacro = caller_location_from_macro!();
    assert_eq!(inmacro.file(), file!());
    assert_eq!(inmacro.line(), 45);
    assert_eq!(inmacro.column(), 19);

    let intrinsic = core::intrinsics::caller_location();
    assert_eq!(intrinsic.file(), file!());
    assert_eq!(intrinsic.line(), 50);
    assert_eq!(intrinsic.column(), 21);
}
