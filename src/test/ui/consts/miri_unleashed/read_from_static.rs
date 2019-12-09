// run-pass
// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(const_mut_refs)]
#![allow(const_err)]

static OH_YES: &mut i32 = &mut 42;

fn main() {
    // Make sure `OH_YES` can be read.
    assert_eq!(*OH_YES, 42);
}
