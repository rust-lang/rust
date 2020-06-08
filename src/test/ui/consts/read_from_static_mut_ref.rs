#![feature(const_mut_refs)]
#![allow(const_err)]

static OH_NO: &mut i32 = &mut 42;
//~^ ERROR references in statics may only refer to immutable values
fn main() {
    assert_eq!(*OH_NO, 42);
}
