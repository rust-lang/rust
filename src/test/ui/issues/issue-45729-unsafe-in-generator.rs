#![feature(generators)]

fn main() {
    let _ = || {
        *(1 as *mut u32) = 42;
        //~^ ERROR dereference of raw pointer is unsafe
        yield;
    };
}
