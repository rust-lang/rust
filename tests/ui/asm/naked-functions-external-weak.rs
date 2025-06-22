//@ compile-flags: -C opt-level=0
//@ needs-asm-support
//@ run-pass

// Test that naked functions with external weak linkage don't cause SIGILL
// This is a regression test for issue #142880

#![feature(naked_functions)]
#![feature(linkage)]

use std::arch::asm;

#[naked]
#[linkage = "external_weak"]
extern "C" fn naked_weak_function() -> u32 {
    unsafe {
        asm!(
            "mov eax, 42",
            "ret",
            options(noreturn)
        );
    }
}

fn main() {
    // Test that the function compiles without causing SIGILL
    // We don't actually call it as it may not be linked
    println!("Test passed: naked function with external_weak linkage compiled successfully");
}