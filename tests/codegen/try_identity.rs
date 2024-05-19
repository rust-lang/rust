//@ compile-flags: -C no-prepopulate-passes -O -Z mir-opt-level=3 -Zunsound-mir-opts

// Ensure that `x?` has no overhead on `Result<T, E>` due to identity `match`es in lowering.
// This requires inlining to trigger the MIR optimizations in `SimplifyArmIdentity`.

#![crate_type = "lib"]

type R = Result<u64, i32>;

// This was written to the `?` from `try_trait`, but `try_trait_v2` uses a different structure,
// so the relevant desugar is copied inline in order to keep the test testing the same thing.
// FIXME(#85133): while this might be useful for `r#try!`, it would be nice to have a MIR
// optimization that picks up the `?` desugaring, as `SimplifyArmIdentity` does not.
#[no_mangle]
pub fn try_identity(x: R) -> R {
// CHECK: start:
// FIXME(JakobDegen): Broken by deaggregation change CHECK-NOT\: br {{.*}}
// CHECK ret void
    let y = match into_result(x) {
        Err(e) => return from_error(From::from(e)),
        Ok(v) => v,
    };
    Ok(y)
}

#[inline]
fn into_result<T, E>(r: Result<T, E>) -> Result<T, E> {
    r
}

#[inline]
fn from_error<T, E>(e: E) -> Result<T, E> {
    Err(e)
}
