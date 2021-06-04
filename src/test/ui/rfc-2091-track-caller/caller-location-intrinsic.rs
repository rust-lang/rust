// run-pass
// revisions: default mir-opt
//[mir-opt] compile-flags: -Zmir-opt-level=4

#[inline(never)]
#[track_caller]
fn codegen_caller_loc() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

macro_rules! caller_location_from_macro {
    () => (codegen_caller_loc());
}

fn main() {
    let loc = codegen_caller_loc();
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
