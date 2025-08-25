// Make sure we catch this even without Stacked Borrows
//@compile-flags: -Zmiri-disable-stacked-borrows

#![allow(integer_to_ptr_transmutes)]

use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR: encountered a dangling reference
}
