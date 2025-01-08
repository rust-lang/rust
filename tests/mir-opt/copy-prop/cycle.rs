// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that cyclic assignments don't hang CopyProp, and result in reasonable code.
//@ test-mir-pass: CopyProp
fn val() -> i32 {
    1
}

// EMIT_MIR cycle.main.CopyProp.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: debug z => [[y]];
    // CHECK-NOT: StorageLive([[y]]);
    // CHECK: [[y]] = copy [[x]];
    // CHECK-NOT: StorageLive(_3);
    // CHECK-NOT: _3 = copy [[y]];
    // CHECK-NOT: StorageLive(_4);
    // CHECK-NOT: _4 = copy _3;
    // CHECK-NOT: _1 = move _4;
    // CHECK: [[x]] = copy [[y]];
    let mut x = val();
    let y = x;
    let z = y;
    x = z;

    drop(x);
}
