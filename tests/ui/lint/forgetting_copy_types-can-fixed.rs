//@ check-fail
//@ run-rustfix

#![deny(forgetting_copy_types)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use std::vec::Vec;
use std::mem::forget;

#[derive(Copy, Clone)]
struct SomeStruct;

fn main() {
    let s1 = SomeStruct {};
    let s2 = s1;
    let mut s3 = s1;

    forget(s1); //~ ERROR calls to `std::mem::forget`
    forget(s2); //~ ERROR calls to `std::mem::forget`
    forget(s3); //~ ERROR calls to `std::mem::forget`
}
