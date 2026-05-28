//@ check-pass
#![allow(dead_code)]
#![allow(unreachable_code)]
// Regression test for issue #39984.
//
// The key here is that the error type of the `Ok` call ought to be
// constrained to `String`, even though it is dead-code.

fn main() {}

fn t() -> Result<(), String> {
    return Err("".into());
    Ok(())
}
