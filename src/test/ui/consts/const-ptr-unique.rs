#![feature(ptr_internals)]

use std::ptr::Unique;

fn main() {
    let mut i: u32 = 10;
    let unique = Unique::new(&mut i).unwrap();
    let x: &'static *mut u32 = &(unique.as_ptr());
    //~^ ERROR temporary value dropped while borrowed
}
