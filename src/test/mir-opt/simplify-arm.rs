// compile-flags: -Z mir-opt-level=1
// EMIT_MIR rustc.id.SimplifyArmIdentity.diff
// EMIT_MIR rustc.id.SimplifyBranchSame.diff
// EMIT_MIR rustc.id_result.SimplifyArmIdentity.diff
// EMIT_MIR rustc.id_result.SimplifyBranchSame.diff
// EMIT_MIR rustc.id_try.SimplifyArmIdentity.diff
// EMIT_MIR rustc.id_try.SimplifyBranchSame.diff

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
