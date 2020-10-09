#![crate_type = "lib"]

type T = i32;
type E = u32;

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR match_identity.id_result.MatchIdentitySimplification.diff
pub fn id_result(a: Result<T, E>) -> Result<T, E> {
    match a {
        Ok(x) => Ok(x),
        Err(y) => Err(y),
    }
}

// EMIT_MIR match_identity.flip_flop.MatchBranchSimplification.diff
pub fn flip_flop(a: Result<T, E>) -> Result<E, T> {
    match a {
        Ok(x) => Err(x),
        Err(y) => Ok(y),
    }
}
