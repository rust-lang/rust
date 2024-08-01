//@ run-rustfix

#![feature(unsafe_extern_blocks)]
#![allow(dead_code)]

extern "C" {
    unsafe fn foo(); //~ ERROR items in unadorned `extern` blocks cannot have safety qualifiers
}

fn main() {}
