// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that we bail out when there are multiple assignments to the same local.
//@ test-mir-pass: CopyProp
fn val() -> i32 {
    1
}

fn cond() -> bool {
    true
}

// EMIT_MIR branch.foo.CopyProp.diff
fn foo() -> i32 {
    // CHECK-LABEL: fn foo(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: bb3: {
    // CHECK: [[y]] = copy [[x]];
    // CHECK: bb5: {
    // CHECK: [[y]] = copy [[x]];
    // CHECK: _0 = copy [[y]];
    let x = val();

    let y = if cond() {
        x
    } else {
        val();
        x
    };

    y
}

fn main() {
    foo();
}
