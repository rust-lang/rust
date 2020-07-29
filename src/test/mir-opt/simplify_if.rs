#[inline(never)]
fn noop() {}

// EMIT_MIR simplify_if.main.SimplifyBranches-after-const-prop.diff
fn main() {
    if false {
        noop();
    }
}
