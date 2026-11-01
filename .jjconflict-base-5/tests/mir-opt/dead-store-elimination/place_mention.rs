// Verify that we account for the `PlaceMention` statement as a use of the tuple,
// and don't remove it as a dead store.
//
//@ test-mir-pass: DeadStoreElimination-initial
//@ compile-flags: -Zmir-preserve-ub

// EMIT_MIR place_mention.main.DeadStoreElimination-initial.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK-NOT: PlaceMention(
    // CHECK: [[tmp:_.*]] =
    // CHECK-NEXT: PlaceMention([[tmp:_.*]]);

    let (_, _) = ("Hello", "World");
}
