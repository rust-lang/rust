//@ add-minicore
//@ build-fail
//@ compile-flags: --target armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
//@ ignore-backends: gcc
//@ error-pattern: inline assembly requires more registers than available

// Regression test for issue #90815.
// Ensure that we emit a proper error message when inline assembly requests
// more registers than available, instead of crashing with a SIGSEGV.

#![feature(no_core)]
#![no_core]
#![crate_type = "rlib"]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub unsafe fn exhausted_registers() {
    let r0: u32;
    let r1: u32;
    let r2: u32;
    let r3: u32;
    let r4: u32;
    let r5: u32;
    let r6: u32;
    let r7: u32;
    let r8: u32;
    let r9: u32;
    let r10: u32;
    let r11: u32;
    let r12: u32;
    let r13: u32;
    let r14: u32;
    let r15: u32;

    asm!(
        "mov {0}, r0",
        "mov {1}, r1",
        "mov {2}, r2",
        "mov {3}, r3",
        "mov {4}, r4",
        "mov {5}, r5",
        "mov {6}, r6",
        "mov {7}, r7",
        "mov {8}, r8",
        "mov {9}, r9",
        "mov {10}, r10",
        "mov {11}, r11",
        "mov {12}, r12",
        "mov {13}, r13",
        "mov {14}, r14",
        "mov {15}, r15",
        out(reg) r0,
        out(reg) r1,
        out(reg) r2,
        out(reg) r3,
        out(reg) r4,
        out(reg) r5,
        out(reg) r6,
        out(reg) r7,
        out(reg) r8,
        out(reg) r9,
        out(reg) r10,
        out(reg) r11,
        out(reg) r12,
        out(reg) r13,
        out(reg) r14,
        out(reg) r15,
    );
}
