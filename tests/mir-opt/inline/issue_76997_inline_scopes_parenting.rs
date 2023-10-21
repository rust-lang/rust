// Tests that MIR inliner can handle `SourceScopeData` parenting correctly. (#76997)

// EMIT_MIR issue_76997_inline_scopes_parenting.main.Inline.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: scope 2
    // CHECK-NEXT: debug x
    // CHECK-NEXT: scope 3
    // CHECK-NEXT: debug y
    let f = |x| { let y = x; y };
    f(())
}
