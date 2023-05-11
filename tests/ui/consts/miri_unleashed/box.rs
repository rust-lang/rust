// compile-flags: -Zunleash-the-miri-inside-of-you

use std::mem::ManuallyDrop;

fn main() {}

static TEST_BAD: &mut i32 = {
    &mut *(Box::new(0))
    //~^ ERROR could not evaluate static initializer
    //~| NOTE calling non-const function `Box::<i32>::new`
};
