// This test is taken from https://github.com/rust-lang/rust/issues/70044.
// This test demonstrates how NLL's outlives constraints are flow-insensitive,
// and are wrongly expected to hold outside the inner scope.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

fn main() {
    let mut zero = &mut 0;
    let mut one = 1;

    {
        let mut _r = &mut zero;
        let mut y = &mut one;
        _r = &mut y;
    }

    println!("{}", one); //[nll]~ ERROR: cannot borrow `one` as immutable
    println!("{}", zero);
}
