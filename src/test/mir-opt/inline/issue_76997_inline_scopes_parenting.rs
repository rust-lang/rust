// Tests that MIR inliner can handle `SourceScopeData` parenting correctly. (#76997)

// EMIT_MIR issue_76997_inline_scopes_parenting.main.Inline.after.mir
fn main() {
    let f = |x| { let y = x; y };
    f(())
}
