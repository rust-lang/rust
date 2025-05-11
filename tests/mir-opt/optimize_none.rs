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
pub fn const_branch() -> i32 {
    // CHECK-LABEL: fn const_branch(
    // CHECK: switchInt(const true) -> [0: [[FALSE:bb[0-9]+]], otherwise: [[TRUE:bb[0-9]+]]];
    // CHECK-NEXT: }
    // CHECK: [[FALSE]]: {
    // CHECK-NEXT: _0 = const 0
    // CHECK-NEXT: goto
    // CHECK-NEXT: }
    // CHECK: [[TRUE]]: {
    // CHECK-NEXT: _0 = const 1
    // CHECK-NEXT: goto
    // CHECK-NEXT: }

    if true { 1 } else { 0 }
}

fn main() {}
