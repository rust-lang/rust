// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that cyclic assignments don't hang DestinationPropagation, and result in reasonable code.
//@ test-mir-pass: DestinationPropagation
fn val() -> i32 {
    1
}

// EMIT_MIR cycle.main.DestinationPropagation.diff
fn main() {
    // CHECK-LABEL: main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: [[x]] = val()
    // CHECK-NOT: [[x]] = {{_.*}};
    let mut x = val();
    let y = x;
    let z = y;
    x = z;

    drop(x);
}
