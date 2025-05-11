// New test for #53818: modifying static memory at compile-time is not allowed.
// The test should never compile successfully

use std::cell::UnsafeCell;

static mut FOO: u32 = 42;
static BOO: () = unsafe {
    FOO = 5;
    //~^ ERROR could not evaluate static initializer [E0080]
};

fn main() {}
