// Test that the goto chain starting from bb0 is collapsed.

// EMIT_MIR rustc.main.SimplifyCfg-initial.diff
// EMIT_MIR rustc.main.SimplifyCfg-early-opt.diff
fn main() {
    loop {
        if bar() {
            break;
        }
    }
}

#[inline(never)]
fn bar() -> bool {
    true
}
