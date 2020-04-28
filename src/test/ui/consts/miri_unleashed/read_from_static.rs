// run-pass
// compile-flags: -Zunleash-the-miri-inside-of-you
#![allow(const_err)]

static OH_YES: &mut i32 = &mut 42;
//~^ WARN skipping const checks

fn main() {
    // Make sure `OH_YES` can be read.
    assert_eq!(*OH_YES, 42);
}
