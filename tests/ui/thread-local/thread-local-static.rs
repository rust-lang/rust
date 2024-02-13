// edition:2018

#![feature(thread_local)]
#![feature(const_swap)]
#![allow(static_mut_ref)]

#[thread_local]
static mut STATIC_VAR_2: [u32; 8] = [4; 8];
const fn g(x: &mut [u32; 8]) {
    //~^ ERROR mutable references are not allowed
    std::mem::swap(x, &mut STATIC_VAR_2)
    //~^ ERROR thread-local statics cannot be accessed
    //~| ERROR mutable references are not allowed
    //~| ERROR use of mutable static is unsafe
}

fn main() {}
