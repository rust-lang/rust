// Regression test for https://github.com/rust-lang/rust/issues/150898
//
// This test fails on Apple platforms with versions containing an allocator
// bug. If this fails with SIGABRT, you may need to upgrade your system to
// avoid other unexpected behavior.
//
// Note that the failure is non-deterministic so this test may pass even in
// conditions where the bug is present. That being said, on the author's
// machine this has a 100% rate of hitting the crash on at least a couple
// threads.
//
// The bug was resolved by macOS 26.4.

//@ run-pass
//@ only-aarch64-apple-darwin
//@ compile-flags: -C opt-level=1 -C codegen-units=1

use std::thread;

fn main() {
    // Running with multiple threads substantially increases the change of
    // hitting the bug.
    thread::scope(|s| {
        for _ in 0..10 {
            s.spawn(run);
        }
    });
}

fn run() {
    // This doesn't always fail, so rerun a few times
    for _ in 0..100 {
        unsafe {
            core::arch::asm!(
                "
                // Alloc 18 bytes
                mov x0, #18
                bl  _malloc
                // Save the pointer to x21
                mov x21, x0
                // Alloc 18 bytes again
                mov x0, #18
                bl _malloc
                // Store the contents of `x13` to the second allocation. `x13` is the
                // magic register to cause the crash, other registers work well.
                str x13, [x0]
                // Free the pointers
                bl  _free
                mov x0, x21
                bl  _free
                ",
                out("x0") _,
                out("x13") _,
                out("x21") _,
                clobber_abi("C"),
            )
        }
    }
}
