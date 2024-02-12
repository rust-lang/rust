// compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(thread_local)]
#![allow(static_mut_ref)]

extern "C" {
    static mut DATA: u8;
}

// Make sure we catch accessing extern static.
static TEST_READ: () = {
    unsafe { let _val = DATA; }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE cannot access extern static
};
static TEST_WRITE: () = {
    unsafe { DATA = 0; }
    //~^ ERROR could not evaluate static initializer
    //~| NOTE cannot access extern static
};

// Just creating a reference is fine, as long as we are not reading or writing.
static TEST_REF: &u8 = unsafe { &DATA };

fn main() {}
