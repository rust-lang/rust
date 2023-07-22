// unit-test: UnreachablePropagation
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

enum Empty {}

fn empty() -> Option<Empty> {
    None
}

// EMIT_MIR unreachable.main.UnreachablePropagation.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: bb0: {
    // CHECK: {{_.*}} = empty()
    // CHECK: bb1: {
    // CHECK: [[ne:_.*]] = Ne({{.*}}, const 1_isize);
    // CHECK-NEXT: assume(move [[ne]]);
    // CHECK-NEXT: goto -> bb6;
    // CHECK: bb2: {
    // CHECK-NEXT: unreachable;
    // CHECK: bb3: {
    // CHECK-NEXT: unreachable;
    // CHECK: bb4: {
    // CHECK-NEXT: unreachable;
    // CHECK: bb5: {
    // CHECK-NEXT: unreachable;
    // CHECK: bb6: {
    // CHECK: return;
    if let Some(_x) = empty() {
        let mut _y;

        if true {
            _y = 21;
        } else {
            _y = 42;
        }

        match _x { }
    }
}
