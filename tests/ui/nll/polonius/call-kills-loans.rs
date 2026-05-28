// `Call` terminators can write to a local which has existing loans
// and those need to be killed like a regular assignment to a local.
// This is a simplified version of issue 47680, is correctly accepted
// by NLL but was incorrectly rejected by Polonius because of these
// missing `killed` facts.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius_next polonius
//@ check-pass
//@ [polonius_next] compile-flags: -Z polonius=next
//@ [polonius] compile-flags: -Z polonius

struct Thing;

impl Thing {
    fn next(&mut self) -> &mut Self { unimplemented!() }
}

fn main() {
    let mut temp = &mut Thing;

    loop {
        let v = temp.next();
        temp = v; // accepted by NLL, was incorrectly rejected by Polonius
    }
}
