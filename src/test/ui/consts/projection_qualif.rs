// revisions: stock mut_refs

#![cfg_attr(mut_refs, feature(const_mut_refs))]

use std::cell::Cell;

const FOO: &u32 = {
    let mut a = 42;
    {
        let b: *mut u32 = &mut a; //~ ERROR mutable references are not allowed in constants
        unsafe { *b = 5; } //~ ERROR dereferencing raw pointers in constants
    }
    &{a}
};

fn main() {}
