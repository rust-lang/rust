// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#[inline(never)]
fn noop() {}

// EMIT_MIR simplify_match.main.GVN.diff
fn main() {
    match {
        let x = false;
        x
    } {
        true => noop(),
        false => {}
    }
}
