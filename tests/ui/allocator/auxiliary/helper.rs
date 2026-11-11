//@ no-prefer-dynamic

#![crate_type = "rlib"]
#![no_std]

extern crate alloc;
use alloc::fmt;

pub fn work_with(p: &dyn fmt::Debug) {
    drop(p);
}
