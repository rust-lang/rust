// compile-flags: -Z mir-opt-level=2 -Zunsound-mir-opts
// EMIT_MIR simplify_arm.id.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id.SimplifyBranchSame.diff
// EMIT_MIR simplify_arm.id_result.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id_result.SimplifyBranchSame.diff
// EMIT_MIR simplify_arm.id_try.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id_try.SimplifyBranchSame.diff

fn id(o: Option<u8>) -> Option<u8> {
    match o {
        Some(v) => Some(v),
        None => None,
    }
}

fn id_result(r: Result<u8, i32>) -> Result<u8, i32> {
    match r {
        Ok(x) => Ok(x),
        Err(y) => Err(y),
    }
}

fn id_try(r: Result<u8, i32>) -> Result<u8, i32> {
    let x = r?;
    Ok(x)
}

fn main() {
    id(None);
    id_result(Ok(4));
    id_try(Ok(4));
}
