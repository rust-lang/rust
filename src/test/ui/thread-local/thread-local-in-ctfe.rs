#![feature(const_fn, thread_local)]

#[thread_local]
static A: u32 = 1;

static B: u32 = A;
//~^ ERROR thread-local statics cannot be accessed at compile-time

static C: &u32 = &A;
//~^ ERROR thread-local statics cannot be accessed at compile-time
//~| WARNING thread-local variable borrowed past end of function
//~| WARNING this error has been downgraded to a warning
//~| WARNING this warning will become a hard error in the future

const D: u32 = A;
//~^ ERROR thread-local statics cannot be accessed at compile-time

const E: &u32 = &A;
//~^ ERROR thread-local statics cannot be accessed at compile-time
//~| WARNING thread-local variable borrowed past end of function
//~| WARNING this error has been downgraded to a warning
//~| WARNING this warning will become a hard error in the future

const fn f() -> u32 {
    A
    //~^ ERROR thread-local statics cannot be accessed at compile-time
}

fn main() {}
