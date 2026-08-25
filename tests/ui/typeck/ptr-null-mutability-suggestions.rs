//@ run-rustfix

#[allow(unused_imports)]
use std::ptr;

fn expecting_null_mut(_: *mut u8) {}

fn main() {
    expecting_null_mut(ptr::null());
    //~^ ERROR mismatched types
}
