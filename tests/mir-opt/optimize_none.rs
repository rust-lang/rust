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
    // CHECK: [[BOOL:_[0-9]+]] = const true;
    // CHECK: switchInt(move [[BOOL]]) -> [0: [[BB_FALSE:bb[0-9]+]], otherwise: [[BB_TRUE:bb[0-9]+]]];
    // CHECK-NEXT: }
    // CHECK: [[BB_FALSE]]: {
    // CHECK-NEXT: _0 = const 0
    // CHECK-NEXT: goto
    // CHECK-NEXT: }
    // CHECK: [[BB_TRUE]]: {
    // CHECK-NEXT: _0 = const 1
    // CHECK-NEXT: goto
    // CHECK-NEXT: }

    if true { 1 } else { 0 }
}

fn main() {}
