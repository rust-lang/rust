// Verifies that calling a function pointer with a mismatched type triggers a
// CFI violation and causes the process to trap.

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [cfi] needs-sanitizer-support
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer
//@ [cfi] compile-flags: -C opt-level=0 -C codegen-units=1 -C lto
//@ [cfi] compile-flags: -C prefer-dynamic=off
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [cfi] compile-flags: -Z sanitizer-cfi-diag=true
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ [kcfi] compile-flags: -C panic=abort -C prefer-dynamic=off
//@ run-fail-or-crash

use std::hint::black_box;
use std::mem;

fn add_one(x: i32) -> i32 {
    x + 1
}

// Accept a function pointer as a parameter so that the indirect call cannot
// be devirtualized by the compiler.
#[inline(never)]
fn call_with_mismatch(f: fn(i32) -> i32) {
    // Transmute fn(i32) -> i32 into fn(i32, i32) -> i32, creating a
    // function pointer type mismatch that CFI should catch.
    let g: fn(i32, i32) -> i32 = unsafe { mem::transmute(f) };
    // This indirect call should fail the CFI type check and trap.
    let _result = g(1, 2);
}

fn main() {
    call_with_mismatch(black_box(add_one));
}
