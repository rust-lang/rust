// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#[inline(never)]
fn noop() {}

// EMIT_MIR simplify_if.main.SimplifyConstCondition-after-inst-simplify.diff
fn main() {
    // CHECK-LABEL: fn main(

    // CHECK: bb0: {
    // CHECK-NEXT: return;
    if false {
        noop();
    }
}
