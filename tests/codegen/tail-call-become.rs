//@ compile-flags: -C opt-level=0 -Cpanic=abort -C no-prepopulate-passes
//@ needs-llvm-components: x86

#![feature(explicit_tail_calls)]
#![crate_type = "lib"]

// CHECK-LABEL: define {{.*}}@with_tail(
#[no_mangle]
#[inline(never)]
pub fn with_tail(n: u32) -> u32 {
    // CHECK: tail call {{.*}}@with_tail(
    if n == 0 { 0 } else { become with_tail(n - 1) }
}

// CHECK-LABEL: define {{.*}}@no_tail(
#[no_mangle]
#[inline(never)]
pub fn no_tail(n: u32) -> u32 {
    // CHECK-NOT: tail call
    // CHECK: call {{.*}}@no_tail(
    if n == 0 { 0 } else { no_tail(n - 1) }
}

// CHECK-LABEL: define {{.*}}@even_with_tail(
#[no_mangle]
#[inline(never)]
pub fn even_with_tail(n: u32) -> bool {
    // CHECK: tail call {{.*}}@odd_with_tail(
    match n {
        0 => true,
        _ => become odd_with_tail(n - 1),
    }
}

// CHECK-LABEL: define {{.*}}@odd_with_tail(
#[no_mangle]
#[inline(never)]
pub fn odd_with_tail(n: u32) -> bool {
    // CHECK: tail call {{.*}}@even_with_tail(
    match n {
        0 => false,
        _ => become even_with_tail(n - 1),
    }
}

// CHECK-LABEL: define {{.*}}@even_no_tail(
#[no_mangle]
#[inline(never)]
pub fn even_no_tail(n: u32) -> bool {
    // CHECK-NOT: tail call
    // CHECK: call {{.*}}@odd_no_tail(
    match n {
        0 => true,
        _ => odd_no_tail(n - 1),
    }
}

// CHECK-LABEL: define {{.*}}@odd_no_tail(
#[no_mangle]
#[inline(never)]
pub fn odd_no_tail(n: u32) -> bool {
    // CHECK-NOT: tail call
    // CHECK: call {{.*}}@even_no_tail(
    match n {
        0 => false,
        _ => even_no_tail(n - 1),
    }
}
