//@ run-rustfix

#![allow(dead_code)]

extern "C" {
    unsafe fn foo(); //~ ERROR items in `extern` blocks without an `unsafe` qualifier cannot have safety qualifiers
}

fn main() {}
