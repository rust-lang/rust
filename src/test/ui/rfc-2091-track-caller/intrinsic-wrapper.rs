// run-pass

macro_rules! caller_location_from_macro {
    () => (core::panic::Location::caller());
}

fn main() {
    let loc = core::panic::Location::caller();
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), 8);
    assert_eq!(loc.column(), 15);

    // `Location::caller()` in a macro should behave similarly to `file!` and `line!`,
    // i.e. point to where the macro was invoked, instead of the macro itself.
    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.file(), file!());
    assert_eq!(loc2.line(), 15);
    assert_eq!(loc2.column(), 16);
}
