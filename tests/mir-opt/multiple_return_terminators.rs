//@ test-mir-pass: MultipleReturnTerminators
//@ compile-flags: -Z mir-enable-passes=+SimplifyCfg-final

// EMIT_MIR multiple_return_terminators.test.MultipleReturnTerminators.diff
// EMIT_MIR multiple_return_terminators.test.SimplifyCfg-final.diff

#[inline(never)]
fn test(x: bool) -> i32 {
    // CHECK-LABEL: fn test

    // CHECK: _0 = const 42_i32;
    // CHECK: return

    // CHECK: _0 = const 2015_i32;
    // CHECK: return

    if x { 42 } else { 2015 }
}

fn main() {
    test(true);
}
