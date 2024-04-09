//@ run-pass
//@ compile-flags:-Cno-prepopulate-passes -Copt-level=0
//@ aux-build:static_fnptr.rs

extern crate static_fnptr;
use static_fnptr::{ADDR, bar};

fn baz() -> bool {
    bar(ADDR)
}

fn main() {
    assert!(baz())
}
