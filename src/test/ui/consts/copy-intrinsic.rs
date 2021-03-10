// ignore-tidy-linelength
#![feature(const_mut_refs, const_intrinsic_copy, const_ptr_offset)]
use std::ptr;

const COPY_ZERO: () = unsafe {
    // Since we are not copying anything, this should be allowed.
    let src = ();
    let mut dst = ();
    ptr::copy_nonoverlapping(&src as *const _ as *const i32, &mut dst as *mut _ as *mut i32, 0);
};

const COPY_OOB_1: () = unsafe {
    let mut x = 0i32;
    let dangle = (&mut x as *mut i32).wrapping_add(10);
    // Even if the first ptr is an int ptr and this is a ZST copy, we should detect dangling 2nd ptrs.
    ptr::copy_nonoverlapping(0x100 as *const i32, dangle, 0); //~ ERROR any use of this value will cause an error
    //~| memory access failed: pointer must be in-bounds
    //~| previously accepted
};
const COPY_OOB_2: () = unsafe {
    let x = 0i32;
    let dangle = (&x as *const i32).wrapping_add(10);
    // Even if the second ptr is an int ptr and this is a ZST copy, we should detect dangling 1st ptrs.
    ptr::copy_nonoverlapping(dangle, 0x100 as *mut i32, 0); //~ ERROR any use of this value will cause an error
    //~| memory access failed: pointer must be in-bounds
    //~| previously accepted
};


fn main() {
}
