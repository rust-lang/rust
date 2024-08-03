//@ run-rustfix

#![allow(dead_code)]

extern "C" {
    unsafe fn foo(); //~ ERROR items in unadorned `extern` blocks cannot have safety qualifiers
}

fn main() {}
