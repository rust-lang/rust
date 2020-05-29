#[inline(never)]
fn noop() {}

// EMIT_MIR rustc.main.SimplifyBranches-after-const-prop.diff
fn main() {
    if false {
        noop();
    }
}
