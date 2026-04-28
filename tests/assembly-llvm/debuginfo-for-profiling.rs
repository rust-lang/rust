// Verify that additional discriminators are emitted for profiling with `-Cdebuginfo-for-profiling`:
//  - 0 discriminators are emitted without the flag in the test below
//  - at least 1 discriminator is emitted with the flag in the test below.
//    Actual count depends on the target
//
//
//@ add-minicore
//@ revisions: DEFAULT-X86 DEFAULT-AARCH64 DEBUGINFO-X86 DEBUGINFO-AARCH64
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=2 -Cdebuginfo=line-tables-only
//@ [DEFAULT-X86] compile-flags: --target=x86_64-unknown-linux-gnu
//@ [DEFAULT-X86] needs-llvm-components: x86
//@ [DEFAULT-AARCH64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [DEFAULT-AARCH64] needs-llvm-components: aarch64
//@ [DEBUGINFO-X86] compile-flags: -Cdebuginfo-for-profiling --target=x86_64-unknown-linux-gnu
//@ [DEBUGINFO-X86] needs-llvm-components: x86
//@ [DEBUGINFO-AARCH64] compile-flags: -Cdebuginfo-for-profiling --target=aarch64-unknown-linux-gnu
//@ [DEBUGINFO-AARCH64] needs-llvm-components: aarch64
// DEFAULT-X86-NOT: discriminator
// DEFAULT-AARCH64-NOT: discriminator
// DEBUGINFO-X86-COUNT-1: discriminator
// DEBUGINFO-AARCH64-COUNT-1: discriminator

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
