// Whenever a `StorageDead` MIR statement destroys a value `x`,
// we should kill all loans of `x`. This is extracted from `rand 0.4.6`,
// is correctly accepted by NLL but was incorrectly rejected by
// Polonius because of these missing `killed` facts.

// build-pass
// compile-flags: -Z borrowck=mir -Z polonius
// ignore-compare-mode-nll

use std::{io, mem};
use std::io::Read;

fn fill(r: &mut Read, mut buf: &mut [u8]) -> io::Result<()> {
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
