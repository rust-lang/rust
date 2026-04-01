// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: CopyProp

// EMIT_MIR issue_107511.main.CopyProp.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug i => [[i:_.*]];
    // CHECK-NOT: StorageLive([[i]]);
    // CHECK-NOT: StorageDead([[i]]);
    let mut sum = 0;
    let a = [0, 10, 20, 30];

    // `i` is assigned in a loop. Only removing its `StorageDead` would mean that
    // execution sees repeated `StorageLive`. This would be UB.
    for i in 0..a.len() {
        sum += a[i];
    }
}
