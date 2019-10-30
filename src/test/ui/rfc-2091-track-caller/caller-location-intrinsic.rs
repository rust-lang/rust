// run-pass

#![feature(core_intrinsics)]

macro_rules! caller_location_from_macro {
    () => (core::intrinsics::caller_location());
}

fn main() {
    let loc = core::intrinsics::caller_location();
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), 10);
    assert_eq!(loc.column(), 15);

    // `caller_location()` in a macro should behave similarly to `file!` and `line!`,
    // i.e. point to where the macro was invoked, instead of the macro itself.
    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.file(), file!());
    assert_eq!(loc2.line(), 17);
    assert_eq!(loc2.column(), 16);
}
