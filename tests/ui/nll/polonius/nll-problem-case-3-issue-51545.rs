// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #51545
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

fn borrow(o: &mut Option<i32>) -> Option<&mut i32> {
    match o.as_mut() {
        Some(i) => Some(i),
        None => o.as_mut(),
    }
}

fn main() {
    let mut o: Option<i32> = Some(1i32);

    let x = match o.as_mut() {
        Some(i) => Some(i),
        None => o.as_mut(),
    };
}
