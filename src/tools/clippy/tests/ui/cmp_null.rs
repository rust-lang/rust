#![warn(clippy::cmp_null)]
#![allow(unused_mut)]

use std::ptr;

fn main() {
    let x = 0;
    let p: *const usize = &x;
    if p == ptr::null() {
        println!("This is surprising!");
    }
    let mut y = 0;
    let mut m: *mut usize = &mut y;
    if m == ptr::null_mut() {
        println!("This is surprising, too!");
    }
}
