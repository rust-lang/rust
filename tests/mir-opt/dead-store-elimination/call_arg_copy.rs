// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: DeadStoreElimination
// compile-flags: -Zmir-enable-passes=+CopyProp

#[inline(never)]
fn use_both(_: i32, _: i32) {}

// EMIT_MIR call_arg_copy.move_simple.DeadStoreElimination.diff
fn move_simple(x: i32) {
    use_both(x, x);
}

fn main() {
    move_simple(1);
}
