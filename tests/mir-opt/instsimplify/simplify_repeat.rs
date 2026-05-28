//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ compile-flags: -C panic=abort
#![crate_type = "lib"]

const MYSTERY: usize = 3_usize.pow(2) - 2_usize.pow(3);

// EMIT_MIR simplify_repeat.repeat_once_to_aggregate.InstSimplify-after-simplifycfg.diff
pub fn repeat_once_to_aggregate<T: Copy>(x: T) -> [T; 1] {
    // CHECK-LABEL: fn repeat_once_to_aggregate(
    // CHECK: debug other => [[OTHER:_[0-9]+]]
    // CHECK-NOT: [move {{_[0-9]+}}; 1]
    // CHECK: [[OTHER]] = [move {{_[0-9]+}}];
    // CHECK-NOT: [move {{_[0-9]+}}; 1]
    // CHECK: _0 = [move {{_[0-9]+}}];
    // CHECK-NOT: [move {{_[0-9]+}}; 1]

    let other = [x; MYSTERY];

    [x; 1]
}
