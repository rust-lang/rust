//@ compile-flags: -Zunleash-the-miri-inside-of-you

use std::mem::ManuallyDrop;

fn main() {}

static TEST_OK: () = {
    let v: Vec<i32> = Vec::new();
    let _v = ManuallyDrop::new(v);
};

// Make sure we catch executing bad drop functions.
// The actual error is tested by the error-pattern above.
static TEST_BAD: () = {
    let _v: Vec<i32> = Vec::new();
}; //~ ERROR could not evaluate static initializer
   //~| NOTE calling non-const function `<Vec<i32> as Drop>::drop`
   //~| NOTE inside `std::ptr::drop_in_place::<Vec<i32>> - shim(Some(Vec<i32>))`

//~? WARN skipping const checks
