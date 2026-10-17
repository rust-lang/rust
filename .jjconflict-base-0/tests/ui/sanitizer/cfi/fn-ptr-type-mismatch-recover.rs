// Verifies that calling a function pointer with a mismatched type with CFI
// recovery enabled reports the failure and continues execution.

//@ revisions: cfi cfi-minimal-runtime
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ [cfi] needs-sanitizer-cfi
//@ [cfi] needs-sanitizer-support
//@ [cfi-minimal-runtime] needs-sanitizer-cfi
//@ [cfi-minimal-runtime] needs-sanitizer-support
//@ compile-flags: -C target-feature=-crt-static
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer,sanitizer-cfi-minimal-runtime
//@ compile-flags: -C opt-level=0 -C codegen-units=1 -C lto
//@ compile-flags: -C prefer-dynamic=off
//@ compile-flags: -Z sanitizer=cfi
//@ [cfi] compile-flags: -Z sanitizer-cfi-recover=true
//@ [cfi-minimal-runtime] compile-flags: -Z sanitizer-cfi-recover=true
//@ [cfi-minimal-runtime] compile-flags: -Z sanitizer-cfi-minimal-runtime=true
//@ run-pass

use std::hint::black_box;
use std::mem;

fn add_one(x: i32) -> i32 {
    x + 1
}

#[inline(never)]
fn call_with_mismatch(f: fn(i32) -> i32) {
    let g: fn(i32, i32) -> i32 = unsafe { mem::transmute(f) };
    let res = g(1, 2);
    assert_eq!(res, 2);
}

fn main() {
    call_with_mismatch(black_box(add_one));
}
