// run-pass
#![allow(unused_variables)]
// Test coercions involving DST and/or raw pointers

// pretty-expanded FIXME #23616

struct S;
trait T { fn dummy(&self) { } }
impl T for S {}

pub fn main() {
    let x: &dyn T = &S;
    // Test we can convert from &-ptr to *-ptr of trait objects
    let x: *const dyn T = &S;

    // Test we can convert from &-ptr to *-ptr of struct pointer (not DST)
    let x: *const S = &S;

    // As above, but mut
    let x: &mut dyn T = &mut S;
    let x: *mut dyn T = &mut S;

    let x: *mut S = &mut S;

    // Test we can change the mutability from mut to const.
    let x: &dyn T = &mut S;
    let x: *const dyn T = &mut S;
}
