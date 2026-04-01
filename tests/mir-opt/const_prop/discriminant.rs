//@ test-mir-pass: GVN

// FIXME(wesleywiser): Ideally, we could const-prop away all of this and just be left with
// `let x = 42` but that doesn't work because const-prop doesn't support `Operand::Indirect`
// and `InterpCx::eval_place()` always forces an allocation which creates the `Indirect`.
// Fixing either of those will allow us to const-prop this away.

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR discriminant.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: bb0: {
    // CHECK: switchInt(const 1_isize) -> [1: bb1, otherwise: bb3];
    // CHECK: bb1: {
    // CHECK: switchInt(const true) -> [0: bb3, otherwise: bb2];
    // CHECK: bb2: {
    // CHECK: [[tmp:_.*]] = const 42_i32;
    // CHECK: goto -> bb4;
    // CHECK: bb3: {
    // CHECK: [[tmp]] = const 10_i32;
    // CHECK: goto -> bb4;
    // CHECK: bb4: {
    // CHECK: {{_.*}} = Add(move [[tmp]], const 0_i32);
    let x = (if let Some(true) = Some(true) { 42 } else { 10 }) + 0;
}
