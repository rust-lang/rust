#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #112087
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

fn issue_112087<'a>(opt: &'a mut Option<i32>, b: bool) -> Result<&'a mut Option<i32>, &'a mut i32> {
    if let Some(v) = opt {
        if b {
            return Err(v);
        }
    }

    *opt = None;
    return Ok(opt);
}
