// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(coroutines)]

fn main() {
    let _ = || {
        *(1 as *mut u32) = 42;
        //~^ ERROR dereference of raw pointer is unsafe
        yield;
    };
}
