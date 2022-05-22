// aux-build:enums.rs
// run-pass

extern crate enums;

use enums::FieldLessWithNonExhaustiveVariant;

fn main() {
    let e = FieldLessWithNonExhaustiveVariant::default();
    // FIXME: https://github.com/rust-lang/rust/issues/91161
    // This `as` cast *should* be an error, since it would fail
    // if the non-exhaustive variant got fields.  But today it
    // doesn't.  The fix for that will update this test to
    // show an error (and not be run-pass any more).
    let d = e as u8;
    assert_eq!(d, 0);
}
