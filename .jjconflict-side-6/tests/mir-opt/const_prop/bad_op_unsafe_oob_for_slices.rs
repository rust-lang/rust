//@ test-mir-pass: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR bad_op_unsafe_oob_for_slices.main.GVN.diff
#[allow(unconditional_panic)]
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug _b => [[b:_.*]];
    // CHECK: [[b]] = copy (*[[a]])[3 of 4];
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
