// aux-build:deprecated-safe.rs
// revisions: mir thir
// NOTE(skippy) these tests output many duplicates, so deduplicate or they become brittle to changes
// [mir]compile-flags: -Zdeduplicate-diagnostics=yes
// [thir]compile-flags: -Z thir-unsafeck -Zdeduplicate-diagnostics=yes

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::{depr_safe, depr_safe_2015, depr_safe_2018};

unsafe fn unsafe_fn() {
    unsafe {} //~ WARN unnecessary `unsafe` block

    depr_safe(); //~ WARN use of function `deprecated_safe::depr_safe` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2015(); //~ ERROR call to unsafe function is unsafe and requires unsafe block
    unsafe {
        depr_safe_2015();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2018(); //~ WARN use of function `deprecated_safe::depr_safe_2018` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_2018();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }
}

fn main() {
    unsafe {} //~ WARN unnecessary `unsafe` block

    depr_safe(); //~ WARN use of function `deprecated_safe::depr_safe` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2015(); //~ ERROR call to unsafe function is unsafe and requires unsafe block
    unsafe {
        depr_safe_2015();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2018(); //~ WARN use of function `deprecated_safe::depr_safe_2018` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_2018();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }
}
