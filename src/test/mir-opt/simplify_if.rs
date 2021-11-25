#[inline(never)]
fn noop() {}

// EMIT_MIR simplify_if.main.SimplifyIfConst.diff
// EMIT_MIR simplify_if.main.SimplifyBranches-initial.diff
// EMIT_MIR simplify_if.main.SimplifyCfg-early-opt.diff
fn main() {
    if false {
        noop();
    }
}
