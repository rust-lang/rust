//@ revisions: NO-OPT SPEED-OPT
//@[NO-OPT] compile-flags: -Copt-level=0
//@[SPEED-OPT] compile-flags: -Copt-level=3 -Coverflow-checks=y

#![feature(optimize_attribute)]

#[optimize(none)]
pub fn add_noopt() -> i32 {
    // CHECK-LABEL: fn add_noopt(
    // CHECK: AddWithOverflow(const 1_i32, const 2_i32);
    // CHECK-NEXT: assert
    1 + 2
}

#[optimize(none)]
#[allow(unused_assignments)]
pub fn dead_store_noopt(input: i32) -> i32 {
    // CHECK-LABEL: fn dead_store_noopt(
    // CHECK: debug value => [[VALUE:_[0-9]+]];
    // CHECK: [[VALUE]] = copy _1;
    // CHECK-NEXT: [[VALUE]] = const 1_i32;
    // CHECK-NEXT: [[VALUE]] = const 2_i32;
    // CHECK-NEXT: _0 = copy [[VALUE]];

    let mut value = input;
    value = 1;
    value = 2;
    value
}

fn main() {}
