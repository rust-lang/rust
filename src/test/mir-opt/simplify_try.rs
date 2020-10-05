// compile-flags: -Zunsound-mir-opts
// EMIT_MIR simplify_try.try_identity.SimplifyArmIdentity.diff
// EMIT_MIR simplify_try.try_identity.SimplifyBranchSame.after.mir
// EMIT_MIR simplify_try.try_identity.SimplifyLocals.after.mir
// EMIT_MIR simplify_try.try_identity.DestinationPropagation.diff

fn try_identity(x: Result<u32, i32>) -> Result<u32, i32> {
    let y = x?;
    Ok(y)
}

fn main() {
    let _ = try_identity(Ok(0));
}
