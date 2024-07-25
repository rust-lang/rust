// This example is interesting because the non-transitive version of `MaybeLiveLocals` would
// report that *all* of these stores are live.
//
//@ needs-unwind
//@ test-mir-pass: DeadStoreElimination-initial

#[inline(never)]
fn cond() -> bool {
    false
}

// EMIT_MIR cycle.cycle.DeadStoreElimination-initial.diff
fn cycle(mut x: i32, mut y: i32, mut z: i32) {
    // CHECK-LABEL: fn cycle(
    // CHECK-NOT: {{_.*}} =
    // CHECK: {{_.*}} = cond()
    // CHECK-NOT: {{_.*}} =
    // CHECK: _0 = const ();
    // CHECK-NOT: {{_.*}} =
    while cond() {
        let temp = z;
        z = y;
        y = x;
        x = temp;
    }
}

fn main() {
    // CHECK-LABEL: fn main(
    cycle(1, 2, 3);
}
