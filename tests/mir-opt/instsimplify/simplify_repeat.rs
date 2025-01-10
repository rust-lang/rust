//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ compile-flags: -C panic=abort
#![crate_type = "lib"]

// EMIT_MIR simplify_repeat.repeat_once_to_aggregate.InstSimplify-after-simplifycfg.diff
pub fn repeat_once_to_aggregate<T: Copy>(x: T) -> [T; 1] {
    // CHECK-LABEL: fn repeat_once_to_aggregate(
    // CHECK-NOT: [move {{_[0-9]+}}; 1]
    // CHECK: _0 = [move {{_[0-9]+}}];
    // CHECK-NOT: [move {{_[0-9]+}}; 1]

    [x; 1]
}
