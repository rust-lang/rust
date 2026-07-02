//! Test that GVN propagates the constant `false` and eliminates the match.
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#[inline(never)]
fn noop() {}

// EMIT_MIR simplify_match.main.GVN.diff
// CHECK-LABEL: fn main(
// CHECK: debug x => [[x:_.*]];
// CHECK-NOT: switchInt
// CHECK: bb0: {
// CHECK-NEXT: StorageLive([[x]]);
// CHECK-NEXT: [[x]] = const false;
// CHECK-NEXT: StorageDead([[x]]);
// CHECK-NEXT: return;
fn main() {
    match {
        let x = false;
        x
    } {
        true => noop(),
        false => {}
    }
}
