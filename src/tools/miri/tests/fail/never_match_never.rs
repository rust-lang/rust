// This should fail even without validation
//@compile-flags: -Zmiri-disable-validation

#![feature(never_type)]
#![allow(unreachable_code)]

fn main() {
    let ptr: *const (i32, !) = &0i32 as *const i32 as *const _;
    unsafe { match (*ptr).1 {} } //~ ERROR: entering unreachable code
}
