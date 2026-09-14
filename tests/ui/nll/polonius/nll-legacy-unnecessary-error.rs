// NLLs and legacy polonius emit an unnecessary error here, unlike the alpha. It's not clear
// *exactly* why the datalog implementation rejects this, but it looks like it propagates the loan
// from 'x to 'y very eagerly, even though x is dead before the assignment. The loan would thus be
// live and invalidated by the assignment, AKA an error.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] compile-flags: -Z polonius=off
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy

fn main() {
    let mut x: (&u32,) = (&1,);
    let mut y: (&u32,) = (&2,);
    let mut z = 3;

    y.0 = x.0;
    x.0 = &z;
    z += 1;
    //[nll]~^ ERROR: cannot assign to `z` because it is borrowed
    //[legacy]~^^ ERROR: cannot assign to `z` because it is borrowed

    dbg!(y.0);
}
