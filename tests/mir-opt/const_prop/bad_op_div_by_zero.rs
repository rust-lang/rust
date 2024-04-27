//@ test-mir-pass: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// EMIT_MIR bad_op_div_by_zero.main.GVN.diff
#[allow(unconditional_panic)]
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug _z => [[z:_.*]];
    // CHECK: assert(!const true, "attempt to divide `{}` by zero", const 1_i32)
    // CHECK: assert(!const false, "attempt to compute `{} / {}`, which would overflow", const 1_i32, const 0_i32)
    // CHECK: [[z]] = Div(const 1_i32, const 0_i32);
    let y = 0;
    let _z = 1 / y;
}
