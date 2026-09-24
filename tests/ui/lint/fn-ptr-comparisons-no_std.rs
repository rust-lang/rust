//@ check-pass
//@ run-rustfix

#![no_std]
#![crate_type = "lib"]
#![allow(dead_code)]

fn a() {}

fn cmp(f: fn()) -> bool {
    f == a
    //~^ WARN function pointer comparisons
}
