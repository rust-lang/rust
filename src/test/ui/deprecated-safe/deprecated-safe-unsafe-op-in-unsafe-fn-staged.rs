// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(deprecated_safe_in_future, unused_unsafe)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate deprecated_safe;

use deprecated_safe::{
    depr_safe, depr_safe_2015, depr_safe_2015_future, depr_safe_2018, depr_safe_future,
};

unsafe fn unsafe_fn() {
    unsafe {} //~ WARN unnecessary `unsafe` block

    depr_safe(); //~ WARN use of function `deprecated_safe::depr_safe` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_future(); //~ WARN use of function `deprecated_safe::depr_safe_future` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_future();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2015(); //~ ERROR call to unsafe function is unsafe and requires unsafe block
    unsafe {
        depr_safe_2015();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2015_future(); //~ WARN use of function `deprecated_safe::depr_safe_2015_future` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_2015_future();
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

    depr_safe_future(); //~ WARN use of function `deprecated_safe::depr_safe_future` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_future();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2015(); //~ ERROR call to unsafe function is unsafe and requires unsafe block
    unsafe {
        depr_safe_2015();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2015_future(); //~ WARN use of function `deprecated_safe::depr_safe_2015_future` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_2015_future();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }

    depr_safe_2018(); //~ WARN use of function `deprecated_safe::depr_safe_2018` without an unsafe block has been deprecated as it is now an unsafe function
    unsafe {
        depr_safe_2018();
        unsafe {} //~ WARN unnecessary `unsafe` block
    }
}
