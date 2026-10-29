// Checks that code coverage doesn't interfere with const_precise_live_drops.
// Regression test for issue #93848.
//
//@ check-pass
//@ compile-flags: --crate-type=lib -Cinstrument-coverage  -Zno-profiler-runtime

#![feature(const_precise_live_drops)]

#[inline]
pub const fn transpose<T, E>(this: Option<Result<T, E>>) -> Result<Option<T>, E> {
    match this {
        Some(Ok(x)) => Ok(Some(x)),
        Some(Err(e)) => Err(e),
        None => Ok(None),
    }
}
