//@ test-mir-pass: UnreachablePropagation
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

pub enum Empty {}

fn empty() -> Option<Empty> {
    None
}

fn loop_forever() {
    loop {}
}

// EMIT_MIR unreachable_diverging.main.UnreachablePropagation.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: bb0: {
    // CHECK: {{_.*}} = empty()
    // CHECK: bb1: {
    // CHECK: switchInt({{.*}}) -> [1: bb2, otherwise: bb6];
    // CHECK: bb2: {
    // CHECK: [[ne:_.*]] = Ne({{.*}}, const false);
    // CHECK: assume(move [[ne]]);
    // CHECK: goto -> bb3;
    // CHECK: bb3: {
    // CHECK: {{_.*}} = loop_forever()
    // CHECK: bb4: {
    // CHECK: unreachable;
    // CHECK: bb5: {
    // CHECK: unreachable;
    // CHECK: bb6: {
    // CHECK: return;
    let x = true;
    if let Some(bomb) = empty() {
        if x {
            loop_forever()
        }
        match bomb {}
    }
}
