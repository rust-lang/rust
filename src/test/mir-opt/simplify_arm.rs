// compile-flags: -Zunsound-mir-opts

// EMIT_MIR simplify_arm.id.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id.SimplifyBranchSame.diff
fn id(o: Option<u8>) -> Option<u8> {
    match o {
        Some(v) => Some(v),
        None => None,
    }
}

// EMIT_MIR simplify_arm.id_result.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id_result.SimplifyBranchSame.diff
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
// EMIT_MIR simplify_arm.id_try.SimplifyArmIdentity.diff
// EMIT_MIR simplify_arm.id_try.SimplifyBranchSame.diff
fn id_try(r: Result<u8, i32>) -> Result<u8, i32> {
    let x = match into_result(r) {
        Err(e) => return from_error(From::from(e)),
        Ok(v) => v,
    };
    Ok(x)
}

#[derive(Copy, Clone)]
enum MultiField {
    A(u32, u32),
    B(u32, u32, u32),
}

// EMIT_MIR simplify_arm.multi_field_id.SimplifyArmIdentity.diff
fn multi_field_id(x: MultiField) -> MultiField {
    match x {
        MultiField::A(a, b) => MultiField::A(a, b),
        MultiField::B(a, b, c) => MultiField::B(a, b, c),
    }
}

enum NonCopy {
    A(String),
    B(Vec<u32>),
}

// EMIT_MIR simplify_arm.non_copy_id.SimplifyArmIdentity.diff
fn non_copy_id(x: NonCopy) -> NonCopy {
    match x {
        NonCopy::A(a) => NonCopy::A(a),
        NonCopy::B(b) => NonCopy::B(b),
    }
}

fn main() {
    id(None);
    id_result(Ok(4));
    id_try(Ok(4));
    multi_field_id(MultiField::A(0, 0));
    non_copy_id(NonCopy::A(String::new()));
}
