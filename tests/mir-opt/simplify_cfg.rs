// skip-filecheck
// Test that the goto chain starting from bb0 is collapsed.
// compile-flags: -Cpanic=abort
// no-prefer-dynamic

// EMIT_MIR simplify_cfg.main.SimplifyCfg-initial.diff
// EMIT_MIR simplify_cfg.main.SimplifyCfg-early-opt.diff
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
