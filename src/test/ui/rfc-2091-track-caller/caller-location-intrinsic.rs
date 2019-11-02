// run-pass

#![feature(core_intrinsics)]
fn main() {
    let loc = core::intrinsics::caller_location();
    assert_eq!(loc.file(), file!());
    assert_eq!(loc.line(), 5);
    assert_eq!(loc.column(), 15);
}
