#![feature(thread_local)]

#[thread_local]
static A: u32 = 1;

static B: u32 = A;
//~^ ERROR thread-local statics cannot be accessed at compile-time

static C: &u32 = &A;
//~^ ERROR thread-local statics cannot be accessed at compile-time

const D: u32 = A;
//~^ ERROR thread-local statics cannot be accessed at compile-time

const E: &u32 = &A;
//~^ ERROR thread-local statics cannot be accessed at compile-time

const fn f() -> u32 {
    A
    //~^ ERROR thread-local statics cannot be accessed at compile-time
}

fn main() {}
