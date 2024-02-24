//@ unit-test: UnreachablePropagation
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

enum Empty {}

fn empty() -> Option<Empty> {
    None
}

// EMIT_MIR unreachable.if_let.UnreachablePropagation.diff
fn if_let() {
    // CHECK-LABEL: fn if_let(
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

// EMIT_MIR unreachable.as_match.UnreachablePropagation.diff
fn as_match() {
    // CHECK-LABEL: fn as_match(
    // CHECK: bb0: {
    // CHECK: {{_.*}} = empty()
    // CHECK: bb1: {
    // CHECK: [[eq:_.*]] = Eq({{.*}}, const 0_isize);
    // CHECK-NEXT: assume(move [[eq]]);
    // CHECK-NEXT: goto -> bb4;
    // CHECK: bb2: {
    // CHECK-NEXT: unreachable;
    // CHECK: bb3: {
    // CHECK-NEXT: unreachable;
    // CHECK: bb4: {
    // CHECK: return;
    match empty() {
        None => {}
        Some(_x) => match _x {}
    }
}

fn main() {
    if_let();
    as_match();
}
