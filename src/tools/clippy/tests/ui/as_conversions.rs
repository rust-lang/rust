//@aux-build:proc_macros.rs

#![warn(clippy::as_conversions)]
#![allow(clippy::borrow_as_ptr, unused)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn main() {
    let i = 0u32 as u64; //~ as_conversions

    let j = &i as *const u64 as *mut u64;
    //~^ as_conversions
    //~| as_conversions

    external!(0u32 as u64);
}

with_span!(
    span

    fn converting() {
        let x = 0u32 as u64;
    }
);
