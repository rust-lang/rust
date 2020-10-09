// EMIT_MIR match_identity.id_result.match_identity.diff
pub fn id_result(a: Result<u64, i64>) -> Result<u64, i64> {
    match a {
        Ok(x) => Ok(x),
        Err(y) => Err(y),
    }
}

// EMIT_MIR match_identity.id_result.match_identity.diff
pub fn flip_flop(a: Result<u64, i64>) -> Result<i64, u64> {
    match a {
        Ok(x) => Err(x),
        Err(y) => Ok(y),
    }
}
