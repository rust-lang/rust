//@ compile-flags: -Zunleash-the-miri-inside-of-you

use std::mem::ManuallyDrop;


struct NotConstDestruct;

impl Drop for NotConstDestruct {
    fn drop(&mut self) {}
}

fn main() {}

static TEST_OK: () = {
    let v: NotConstDestruct = NotConstDestruct;
    let _v = ManuallyDrop::new(v);
};

// Make sure we catch executing bad drop functions.
// The actual error is tested by the error-pattern above.
static TEST_BAD: () = {
    let _v: NotConstDestruct = NotConstDestruct;
}; //~ NOTE failed inside this call
   //~| ERROR calling non-const function `<NotConstDestruct as Drop>::drop`
   //~| NOTE inside `drop_in_place::<NotConstDestruct> - shim(Some(NotConstDestruct))`

//~? WARN skipping const checks
