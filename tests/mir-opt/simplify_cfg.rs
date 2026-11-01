// Test that the goto chain starting from bb0 is collapsed.
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic

// EMIT_MIR simplify_cfg.main.SimplifyCfg-initial.diff
// EMIT_MIR simplify_cfg.main.SimplifyCfg-post-analysis.diff
//
// CHECK: bb0: {
// CHECK-NEXT: [[LOOP:bb[0-9]+]]
// CHECK: [[LOOP]]: {
// CHECK-NOT: }
// CHECK: [[BAR_RETURN_VALUE:_[0-9]+]] = bar() -> [return: [[BAR_JUMP_BLOCK:bb[0-9]+]]
// CHECK: [[BAR_JUMP_BLOCK]]: {
// CHECK-NEXT: switchInt(move [[BAR_RETURN_VALUE]]) -> [0: [[FALSE_BRANCH:bb[0-9]+]], otherwise: [[TRUE_BRANCH:bb[0-9]+]]]
// CHECK: [[FALSE_BRANCH]]: {
// CHECK-NOT: }
// CHECK: goto -> [[LOOP]];
//
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
