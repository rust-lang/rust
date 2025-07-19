// Verifies that KCFI arity indicator is emitted.
//
//@ add-core-stubs
//@ revisions: x86_64
//@ assembly-output: emit-asm
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu -Cllvm-args=-x86-asm-syntax=intel -Ctarget-feature=-crt-static -Cpanic=abort -Zsanitizer=kcfi -Zsanitizer-kcfi-arity -Copt-level=0
//@ [x86_64] needs-llvm-components: x86
//@ min-llvm-version: 21.0.0

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;

unsafe extern "C" {
    safe fn add(x: i32, y: i32) -> i32;
}

pub fn add_one(x: i32) -> i32 {
    // CHECK-LABEL: __cfi__{{.*}}7add_one{{.*}}:
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  mov ecx, 2628068948
    add(x, 1)
}

pub fn add_two(x: i32, _y: i32) -> i32 {
    // CHECK-LABEL: __cfi__{{.*}}7add_two{{.*}}:
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  mov edx, 2505940310
    add(x, 2)
}

pub fn do_twice(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: __cfi__{{.*}}8do_twice{{.*}}:
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  nop
    // CHECK-NEXT:  mov edx, 653723426
    add(f(arg), f(arg))
}
