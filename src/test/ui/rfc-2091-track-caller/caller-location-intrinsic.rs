// run-pass

#![feature(track_caller)]

#[inline(never)]
#[track_caller]
fn defeat_const_prop() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

macro_rules! caller_location_from_macro {
    () => (defeat_const_prop());
}

fn main() {
    let loc = defeat_const_prop();
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), 16);
    assert_eq!(loc.column(), 15);

    // `Location::caller()` in a macro should behave similarly to `file!` and `line!`,
    // i.e. point to where the macro was invoked, instead of the macro itself.
    let loc2 = caller_location_from_macro!();
    assert_eq!(loc2.file(), file!());
    assert_eq!(loc2.line(), 23);
    assert_eq!(loc2.column(), 16);
}
