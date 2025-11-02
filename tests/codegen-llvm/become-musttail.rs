//@ compile-flags: -C opt-level=0 -Cpanic=abort -C no-prepopulate-passes
//@ needs-unwind

#![crate_type = "lib"]
#![feature(explicit_tail_calls)]

// CHECK-LABEL: define {{.*}}@fibonacci(
#[no_mangle]
#[inline(never)]
pub fn fibonacci(n: u64, a: u64, b: u64) -> u64 {
    // CHECK: musttail call {{.*}}@fibonacci(
    // CHECK-NEXT: ret i64
    match n {
        0 => a,
        1 => b,
        _ => become fibonacci(n - 1, b, a + b),
    }
}
