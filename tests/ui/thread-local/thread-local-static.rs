// edition:2018

#![feature(thread_local)]
#![feature(const_swap)]

#[thread_local]
static mut STATIC_VAR_2: [u32; 8] = [4; 8];
const fn g(x: &mut [u32; 8]) {
    //~^ ERROR mutable references are not allowed
    std::mem::swap(x, &mut STATIC_VAR_2)
    //~^ WARN creating a mutable reference to mutable static is discouraged [static_mut_refs]
    //~^^ ERROR thread-local statics cannot be accessed
    //~| ERROR mutable references are not allowed
    //~| ERROR use of mutable static is unsafe
    //~| constant functions cannot refer to statics
}

fn main() {}
