// EMIT_MIR rustc.try_identity.SimplifyArmIdentity.diff
// EMIT_MIR rustc.try_identity.SimplifyBranchSame.after.mir
// EMIT_MIR rustc.try_identity.SimplifyLocals.after.mir

fn try_identity(x: Result<u32, i32>) -> Result<u32, i32> {
    let y = x?;
    Ok(y)
}

fn main() {
    let _ = try_identity(Ok(0));
}
