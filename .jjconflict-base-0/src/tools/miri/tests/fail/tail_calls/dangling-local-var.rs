#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

fn g(x: *const i32) {
    let _val = unsafe { *x }; //~ERROR: has been freed, so this pointer is dangling
}

fn f(_x: *const i32) {
    let local = 0;
    let ptr = &local as *const i32;
    become g(ptr)
}

fn main() {
    f(std::ptr::null());
}
