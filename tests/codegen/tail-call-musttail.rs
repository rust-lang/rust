//@ compile-flags: -C opt-level=0 -Cpanic=abort -C no-prepopulate-passes
//@ needs-unwind

#![crate_type = "lib"]
#![feature(explicit_tail_calls)]

// Ensure that explicit tail calls use musttail in LLVM

// CHECK-LABEL: define {{.*}}@simple_tail_call(
#[no_mangle]
#[inline(never)]
pub fn simple_tail_call(n: i32) -> i32 {
    // CHECK: musttail call {{.*}}@simple_tail_call(
    // CHECK-NEXT: ret i32
    if n <= 0 {
        0
    } else {
        become simple_tail_call(n - 1)
    }
}

// CHECK-LABEL: define {{.*}}@tail_call_with_args(
#[no_mangle]
#[inline(never)]
pub fn tail_call_with_args(a: i32, b: i32, c: i32) -> i32 {
    // CHECK: musttail call {{.*}}@tail_call_with_args(
    // CHECK-NEXT: ret i32
    if a == 0 {
        b + c
    } else {
        become tail_call_with_args(a - 1, b + 1, c)
    }
}