//@ check-pass

#![recursion_limit = "5"]
#![allow(unused)]
#![deny(improper_ctypes)]

#[repr(C)]
struct F1(*const ());
#[repr(C)]
struct F2(*const ());
#[repr(C)]
struct F3(*const ());
#[repr(C)]
struct F4(*const ());
#[repr(C)]
struct F5(*const ());
#[repr(C)]
struct F6(*const ());

#[repr(C)]
struct B {
    f1: F1,
    f2: F2,
    f3: F3,
    f4: F4,
    f5: F5,
    f6: F6,
}

extern "C" fn foo(_: B) {}

fn main() {}
