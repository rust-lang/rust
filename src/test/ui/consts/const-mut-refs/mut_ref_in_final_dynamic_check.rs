#![feature(const_mut_refs)]
#![feature(raw_ref_op)]
#![feature(const_raw_ptr_deref)]

// This file checks that our dynamic checks catch things that the static checks miss.
// We do not have static checks for these, because we do not look into function bodies.
// We treat all functions as not returning a mutable reference, because there is no way to
// do that without causing the borrow checker to complain (see the B4/helper test in
// mut_ref_in_final.rs).

const fn helper() -> Option<&'static mut i32> { unsafe {
    // Undefined behaviour (integer as pointer), who doesn't love tests like this.
    // This code never gets executed, because the static checks fail before that.
    Some(&mut *(42 as *mut i32)) //~ ERROR any use of this value will cause an error
    //~| WARN this was previously accepted by the compiler but is being phased out
} }
// The error is an evaluation error and not a validation error, so the error is reported
// directly at the site where it occurs.
const A: Option<&mut i32> = helper();

const fn helper2() -> Option<&'static mut i32> { unsafe {
    // Undefined behaviour (dangling pointer), who doesn't love tests like this.
    // This code never gets executed, because the static checks fail before that.
    Some(&mut *(&mut 42 as *mut i32))
} }
const B: Option<&mut i32> = helper2(); //~ ERROR encountered dangling pointer in final constant

fn main() {}
