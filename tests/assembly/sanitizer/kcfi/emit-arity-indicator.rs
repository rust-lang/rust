// Verifies that KCFI arity indicator is emitted.
//
//@ add-core-stubs
//@ revisions: x86_64
//@ assembly-output: emit-asm
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel -Ctarget-feature=-crt-static -Zsanitizer=kcfi -Zsanitizer-kcfi-arity -Copt-level=0
//@[x86_64] needs-llvm-components: x86
//@ min-llvm-version: 20.1.0

#![crate_type = "lib"]

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
    // CHECK-NEXT:  mov eax, 2628068948
    x + 1
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
    // CHECK-NEXT:  mov eax, 2505940310
    x + 2
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
    // CHECK-NEXT:  mov eax, 653723426
    f(arg) + f(arg)
}
