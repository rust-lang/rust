//@ compile-flags: -Z mir-opt-level=4
// EMIT_MIR multiple_return_terminators.test.MultipleReturnTerminators.diff

fn test(x: bool) {
    // CHECK-LABEL: fn test(
    // CHECK: bb0: {
    // CHECK-NEXT: return;
    // CHECK-NEXT: }
    if x {
        // test
    } else {
        // test
    }
}

fn main() {
    test(true)
}
