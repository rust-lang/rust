//@ test-mir-pass: GVN
//@ compile-flags: -Zmir-enable-passes=+SimplifyConstCondition-after-const-prop
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#[inline(never)]
fn foo(_: i32) {}

// EMIT_MIR switch_int.main.GVN.diff
// EMIT_MIR switch_int.main.SimplifyConstCondition-after-const-prop.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: bb0: {
    // CHECK-NOT: switchInt(
    // CHECK: goto -> [[bb:bb.*]];
    // CHECK: [[bb]]: {
    // CHECK-NOT: _0 = foo(const -1_i32)
    // CHECK: _0 = foo(const 0_i32)
    match 1 {
        1 => foo(0),
        _ => foo(-1),
    }
}
