// Similarly to `alias-bounds-loan-escapes`, the alias here has two outlives
// bounds, so we must register a type test rather than an outlives constraint.
// However, in this case, we never actually require that the alias live longer
// than it does. So, we make the liveness of the outlives args longer than
// necessary.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ edition: 2024
//@ revisions: nll polonius
//@ [nll] compile-flags: -Z polonius=off
//@ [polonius] compile-flags: -Z polonius=next
//@ check-pass

fn wrap<'a: 'b, 'b>(x: &'a String) -> impl Copy + 'a + 'b {
    x
}

pub fn test() {
    let x = String::new();
    let y = String::new();
    let mut o = wrap(&x);
    o = wrap(&y);
    drop(x);
    let _ = o;
    drop(y);
}

fn main() {}
