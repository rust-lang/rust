//@ compile-flags: -C debuginfo=full
// Tests that MIR inliner can handle `SourceScopeData` parenting correctly. (#76997)

// EMIT_MIR issue_76997_inline_scopes_parenting.main.Inline.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: scope 1 {
    // CHECK-NEXT: debug f
    // CHECK-NEXT: scope 2 (inlined main::{closure#0}) {
    // CHECK-NEXT: debug x
    // CHECK-NEXT: scope 3 {
    // CHECK-NEXT: debug y
    // CHECK-NEXT: }
    // CHECK-NEXT: }
    // CHECK-NEXT: }
    let f = |x| {
        let y = x;
        y
    };
    f(())
}
