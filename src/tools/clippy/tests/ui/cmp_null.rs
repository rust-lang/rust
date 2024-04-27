#![warn(clippy::cmp_null)]
#![allow(unused_mut)]

use std::ptr;

fn main() {
    let x = 0;
    let p: *const usize = &x;
    if p == ptr::null() {
        //~^ ERROR: comparing with null is better expressed by the `.is_null()` method
        //~| NOTE: `-D clippy::cmp-null` implied by `-D warnings`
        println!("This is surprising!");
    }
    let mut y = 0;
    let mut m: *mut usize = &mut y;
    if m == ptr::null_mut() {
        //~^ ERROR: comparing with null is better expressed by the `.is_null()` method
        println!("This is surprising, too!");
    }
}
