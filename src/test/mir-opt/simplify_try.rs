// compile-flags: -Zunsound-mir-opts
// EMIT_MIR simplify_try.try_identity.SimplifyArmIdentity.diff
// EMIT_MIR simplify_try.try_identity.SimplifyBranchSame.after.mir
// EMIT_MIR simplify_try.try_identity.SimplifyLocals.after.mir
// EMIT_MIR simplify_try.try_identity.DestinationPropagation.diff


fn into_result<T, E>(r: Result<T, E>) -> Result<T, E> {
    r
}

fn from_error<T, E>(e: E) -> Result<T, E> {
    Err(e)
}

// This was written to the `?` from `try_trait`, but `try_trait_v2` uses a different structure,
// so the relevant desugar is copied inline in order to keep the test testing the same thing.
// FIXME(#85133): while this might be useful for `r#try!`, it would be nice to have a MIR
// optimization that picks up the `?` desugaring, as `SimplifyArmIdentity` does not.
fn try_identity(x: Result<u32, i32>) -> Result<u32, i32> {
    let y = match into_result(x) {
        Err(e) => return from_error(From::from(e)),
        Ok(v) => v,
    };
    Ok(y)
}

fn main() {
    let _ = try_identity(Ok(0));
}
