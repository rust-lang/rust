// Verify that additional discriminators are emitted for profiling with `-Cdebuginfo-for-profiling`:
//  - 0 discriminators are emitted without the flag in the test below
//  - at least 1 discriminator is emitted with the flag in the test below
//
//
//@ add-minicore
//@ revisions: DEFAULT DEBUGINFO
//@ compile-flags: -Copt-level=2 -Cdebuginfo=line-tables-only
//@ [DEBUGINFO] compile-flags: -Cdebuginfo-for-profiling
// DEFAULT-NOT: discriminator
// DEBUGINFO-COUNT-1: discriminator

#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    fn add(_x: i32, _y: i32) -> i32;
    fn mul(_x: i32, _y: i32) -> i32;
    fn compute(_x: i32) -> i32;
    fn cond() -> bool;
}

#[no_mangle]
pub fn f(limit: i32) -> i32 {
    unsafe {
        let mut sum = 0;
        let mut i = 1;

        while cond() {
            if cond() {
                sum = add(sum, compute(i));
            } else {
                sum = add(sum, mul(compute(i), 2));
            }
            i = add(i, 1);
        }

        sum
    }
}
