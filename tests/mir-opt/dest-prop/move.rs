// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DestinationPropagation

#[inline(never)]
fn use_both(_: i32, _: i32) {}

// EMIT_MIR move.move_simple.DestinationPropagation.diff
fn move_simple(x: i32) {
    // CHECK-LABEL: fn move_simple(
    // CHECK: use_both(_1, _1)
    use_both(x, x);
}

fn main() {
    move_simple(1);
}
