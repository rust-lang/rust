// compile-flags: -Z mir-opt-level=3 -Zunsound-mir-opts
// EMIT_MIR simplify_arm.id.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id.SimplifyBranchSame.diff
// EMIT_MIR simplify_arm.id_result.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id_result.SimplifyBranchSame.diff
// EMIT_MIR simplify_arm.id_try.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id_try.SimplifyBranchSame.diff

// This pass is broken since deaggregation changed
// ignore-test

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
fn id_try(r: Result<u8, i32>) -> Result<u8, i32> {
    let x = match into_result(r) {
        Err(e) => return from_error(From::from(e)),
        Ok(v) => v,
    };
    Ok(x)
}

fn main() {
    id(None);
    id_result(Ok(4));
    id_try(Ok(4));
}
