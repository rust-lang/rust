//@ add-minicore
//@ compile-flags: -C no-prepopulate-passes --target bpfel-unknown-none
//@ needs-llvm-components: bpf

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;

#[no_mangle]
fn outer_scalar(a: u64) -> u64 {
    inner_scalar(a) as u64
}

// CHECK-LABEL: define {{.*}} i128 @_R{{.*}}inner_scalar(
// CHECK-SAME:   i64{{[^)]*}}
// CHECK:        ret i128
#[inline(never)]
fn inner_scalar(a: u64) -> i128 {
    a as i128
}

struct Aggregate128([u64; 2]);

#[no_mangle]
fn outer_aggregate(a: u64) -> u64 {
    let Aggregate128([first, _]) = inner_aggregate(a);
    first
}

// CHECK-LABEL: define {{.*}} [2 x i64] @_R{{.*}}inner_aggregate(
// CHECK-SAME:   i64{{[^)]*}}
// CHECK:        ret [2 x i64]
#[inline(never)]
fn inner_aggregate(a: u64) -> Aggregate128 {
    Aggregate128([a, 42])
}
