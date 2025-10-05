//@ test-mir-pass: SingleUseConsts
//@ compile-flags: -g -Zinline-mir -Zmir-enable-passes=+InstSimplify-after-simplifycfg,+DeadStoreElimination-final,+DestinationPropagation -Zvalidate-mir
//! Regression test for <https://github.com/rust-lang/rust/issues/33013>.

#![crate_type = "lib"]

// EMIT_MIR single_use_consts.invalid_debuginfo.SingleUseConsts.diff
pub fn invalid_debuginfo() {
    // CHECK-LABEL: fn invalid_debuginfo(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug z => [[z:_.*]];
    // CHECK: StorageLive([[x]]);
    // CHECK-NOT: [[x]]
    // CHECK: // DBG: [[z]] =
    // CHECK: // DBG: [[a]] =
    // CHECK: return
    foo();
}

pub fn foo() {
    // CHECK-LABEL: fn foo(
    bar(&1);
}

fn bar(x: &isize) {
    // CHECK-LABEL: fn bar(
    let a = 1;
    let mut z = x;
    z = &a;
}
