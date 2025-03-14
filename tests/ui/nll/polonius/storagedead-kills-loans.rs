// Whenever a `StorageDead` MIR statement destroys a value `x`,
// we should kill all loans of `x`. This is extracted from `rand 0.4.6`,
// is correctly accepted by NLL but was incorrectly rejected by
// Polonius because of these missing `killed` facts.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius_next polonius
//@ check-pass
//@ [polonius_next] compile-flags: -Z polonius=next
//@ [polonius] compile-flags: -Z polonius

use std::{io, mem};
use std::io::Read;

#[allow(dead_code)]
fn fill(r: &mut dyn Read, mut buf: &mut [u8]) -> io::Result<()> {
    while buf.len() > 0 {
        match r.read(buf).unwrap() {
            0 => return Err(io::Error::new(io::ErrorKind::Other,
                                           "end of file reached")),
            n => buf = &mut mem::replace(&mut buf, &mut [])[n..],
            // ^- Polonius had multiple errors on the previous line (where NLL has none)
            // as it didn't know `buf` was killed here, and would
            // incorrectly reject both the borrow expression, and the assignment.
        }
    }
    Ok(())
}

fn main() {
}
