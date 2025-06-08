#![warn(clippy::cmp_null)]
#![allow(unused_mut)]

use std::ptr;

fn main() {
    let x = 0;
    let p: *const usize = &x;
    if p == ptr::null() {
        //~^ cmp_null

        println!("This is surprising!");
    }
    if ptr::null() == p {
        //~^ cmp_null

        println!("This is surprising!");
    }

    let mut y = 0;
    let mut m: *mut usize = &mut y;
    if m == ptr::null_mut() {
        //~^ cmp_null

        println!("This is surprising, too!");
    }
    if ptr::null_mut() == m {
        //~^ cmp_null

        println!("This is surprising, too!");
    }

    let _ = x as *const () == ptr::null();
    //~^ cmp_null
}

fn issue15010() {
    let f: *mut i32 = std::ptr::null_mut();
    debug_assert!(f != std::ptr::null_mut());
    //~^ cmp_null
}
