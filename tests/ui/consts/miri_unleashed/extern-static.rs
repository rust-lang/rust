//@ compile-flags: -Zunleash-the-miri-inside-of-you
#![feature(thread_local)]
#![allow(static_mut_refs)]

extern "C" {
    static mut DATA: u8;
}

// Make sure we catch accessing extern static.
static TEST_READ: () = {
    unsafe { let _val = DATA; }
    //~^ ERROR cannot access extern static
};
static TEST_WRITE: () = {
    unsafe { DATA = 0; }
    //~^ ERROR cannot access extern static
};

// Just creating a reference is fine, as long as we are not reading or writing.
static TEST_REF: &u8 = unsafe { &DATA };

fn main() {}
