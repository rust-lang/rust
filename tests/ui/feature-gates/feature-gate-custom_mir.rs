#![feature(core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*; //~ custom_mir

#[custom_mir(dialect = "built")] //~ ERROR the `#[custom_mir]` attribute is just used for the Rust test suite
pub fn foo(_x: i32) -> i32 {
    mir! {
        {
            Return() //~ custom_mir
        }
    }
}

fn main() {
    assert_eq!(2, foo(2));
}
