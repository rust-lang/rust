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
}; //~ NOTE evaluation of static initializer failed here
   //~| ERROR calling non-const function `<Vec<i32> as Drop>::drop`
   //~| NOTE inside `drop_in_place::<Vec<i32>> - shim(Some(Vec<i32>))`

//~? WARN skipping const checks
