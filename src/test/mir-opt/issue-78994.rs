#![crate_type = "lib"]

// EMIT_MIR issue_78994.should_opt.SimplifyDebugInfo.diff
// EMIT_MIR issue_78994.should_opt.PreCodegen.after.mir
pub fn should_opt(input: bool) {
    let x = input;
}
