// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DeadStoreElimination

#[inline(never)]
fn cond() -> bool {
    false
}

// EMIT_MIR cycle.cycle.DeadStoreElimination.diff
fn cycle(mut x: i32, mut y: i32, mut z: i32) {
    // This example is interesting because the non-transitive version of `MaybeLiveLocals` would
    // report that *all* of these stores are live.
    while cond() {
        let temp = z;
        z = y;
        y = x;
        x = temp;
    }
}

fn main() {
    cycle(1, 2, 3);
}
