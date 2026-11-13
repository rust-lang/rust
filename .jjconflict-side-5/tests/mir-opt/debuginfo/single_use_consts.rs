//@ test-mir-pass: SingleUseConsts
//@ compile-flags: -g -Zinline-mir -Zmir-enable-passes=+InstSimplify-after-simplifycfg,+DeadStoreElimination-final,+DestinationPropagation -Zvalidate-mir
//! Regression test for <https://github.com/rust-lang/rust/issues/33013>.

#![crate_type = "lib"]

// EMIT_MIR single_use_consts.invalid_debuginfo.SingleUseConsts.diff
pub fn invalid_debuginfo() {
    // CHECK-LABEL: fn invalid_debuginfo(
    // CHEK: debug x => const
    // CHEK: debug z => const
    // CHECK-NOT: DBG
    // CHECK: return
    foo();
}

pub fn foo() {
    bar(&1);
}

fn bar(x: &isize) {
    let a = 1;
    let mut z = x;
    z = &a;
}
