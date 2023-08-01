// stderr-per-bitwidth
#![feature(const_mut_refs)]
#![feature(raw_ref_op)]

// This file checks that our dynamic checks catch things that the static checks miss.
// We do not have static checks for these, because we do not look into function bodies.
// We treat all functions as not returning a mutable reference, because there is no way to
// do that without causing the borrow checker to complain (see the B4/helper test in
// mut_ref_in_final.rs).

const fn helper() -> Option<&'static mut i32> { unsafe {
    // Undefined behaviour (integer as pointer), who doesn't love tests like this.
    Some(&mut *(42 as *mut i32))
} }
// The error is an evaluation error and not a validation error, so the error is reported
// directly at the site where it occurs.
const A: Option<&mut i32> = helper(); //~ ERROR it is undefined behavior to use this value
//~^ encountered mutable reference in a `const`

const fn helper2() -> Option<&'static mut i32> { unsafe {
    // Undefined behaviour (dangling pointer), who doesn't love tests like this.
    Some(&mut *(&mut 42 as *mut i32))
} }
const B: Option<&mut i32> = helper2(); //~ ERROR encountered dangling pointer in final constant

fn main() {}
