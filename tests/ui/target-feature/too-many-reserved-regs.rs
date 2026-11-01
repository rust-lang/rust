//@ build-fail
//@ add-minicore
//@ ignore-backends: gcc
//@ needs-llvm-components: riscv
//@ compile-flags: --target=riscv64gc-unknown-linux-gnu -Copt-level=3 -Cdebuginfo=line-tables-only -Zllvm-target-feature=+reserve-x1,+reserve-x3,+reserve-x4,+reserve-x5,+reserve-x6,+reserve-x7,+reserve-x8,+reserve-x9,+reserve-x11,+reserve-x12,+reserve-x13,+reserve-x14,+reserve-x15,+reserve-x16,+reserve-x17,+reserve-x18,+reserve-x19,+reserve-x20,+reserve-x21,+reserve-x22,+reserve-x23,+reserve-x24,+reserve-x25,+reserve-x26,+reserve-x27,+reserve-x28,+reserve-x29,+reserve-x30,+reserve-x31
//@ normalize-stderr: "error: .*/too-many-reserved-regs.rs" -> "error: too-many-reserved-regs.rs"

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[unsafe(no_mangle)]
pub unsafe fn store_val(dst: *mut u32) {
    *dst = 42;
}

//~? ERROR ran out of registers during register allocation in function 'store_val'
