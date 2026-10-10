// Checks that you can tail call functions with different unsafety
//
//@ check-pass
#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

fn a() {
    unsafe {
        become b();
    }
}

unsafe fn b() {
    become c();
}

fn c() {}

fn main() {}
