// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that cyclic assignments don't hang CopyProp, and result in reasonable code.
//@ test-mir-pass: CopyProp
fn val() -> i32 {
    1
}

// EMIT_MIR cycle.main.CopyProp.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug z => _2;
    // CHECK-NOT: StorageLive(_2);
    // CHECK: _2 = copy _1;
    // CHECK-NOT: StorageLive(_3);
    // CHECK-NOT: _3 = copy _2;
    // CHECK-NOT: StorageLive(_4);
    // CHECK-NOT: _4 = copy _3;
    // CHECK-NOT: _1 = move _4;
    // CHECK: _1 = copy _2;
    let mut x = val();
    let y = x;
    let z = y;
    x = z;

    drop(x);
}
