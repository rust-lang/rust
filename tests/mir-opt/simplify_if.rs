// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+GVN,+SimplifyConstCondition-after-gvn
#[inline(never)]
fn noop() {}

// EMIT_MIR simplify_if.main.SimplifyConstCondition-after-gvn.diff
fn main() {
    if false {
        noop();
    }
}
