// revisions: stock mut_refs
//[mut_refs] check-pass

#![cfg_attr(mut_refs, feature(const_mut_refs))]

use std::cell::Cell;

const FOO: &u32 = {
    let mut a = 42;
    {
        let b: *mut u32 = &mut a; //[stock]~ ERROR mutable references are not allowed in constants
        unsafe { *b = 5; } //[stock]~ ERROR dereferencing raw mutable pointers in constants
    }
    &{a}
};

fn main() {}
