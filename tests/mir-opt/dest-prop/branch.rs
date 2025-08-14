// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that assignment in both branches of an `if` are eliminated.
//@ test-mir-pass: DestinationPropagation
fn val() -> i32 {
    1
}

fn cond() -> bool {
    true
}

// EMIT_MIR branch.foo.DestinationPropagation.diff
fn foo() -> i32 {
    // CHECK-LABEL: fn foo(
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[y]] = val()
    // CHECK-NOT: [[y]] = {{_.*}};
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
