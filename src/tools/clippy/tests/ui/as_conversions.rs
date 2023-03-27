// aux-build:proc_macros.rs

#![warn(clippy::as_conversions)]
#![allow(clippy::borrow_as_ptr)]

extern crate proc_macros;
use proc_macros::external;

fn main() {
    let i = 0u32 as u64;

    let j = &i as *const u64 as *mut u64;

    external!(0u32 as u64);
}
