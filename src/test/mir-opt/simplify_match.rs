#[inline(never)]
fn noop() {}

// EMIT_MIR rustc.main.SimplifyBranches-after-copy-prop.diff
fn main() {
    match { let x = false; x } {
        true => noop(),
        false => {},
    }
}
