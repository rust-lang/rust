//@ run-pass
//@ needs-backends: llvm

#![feature(intrinsics)]

#[rustc_intrinsic(llvm = "llvm.fshl.i64")]
fn fshl(a: i64, b: i64, sh: i64) -> i64;

fn main() {
    let res = fshl(1, 2, 3);
    assert_eq!(res, 8);
}
