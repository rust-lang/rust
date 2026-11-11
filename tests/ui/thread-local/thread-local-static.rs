//@ edition:2018

#![feature(thread_local)]
#![allow(static_mut_refs)]

#[thread_local]
static mut STATIC_VAR_2: [u32; 8] = [4; 8];
const fn g(x: &mut [u32; 8]) {
    std::mem::swap(x, &mut STATIC_VAR_2)
    //~^ ERROR thread-local statics cannot be accessed
    //~| ERROR use of mutable static is unsafe
}

fn main() {}
