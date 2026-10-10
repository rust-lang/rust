//@ run-pass
// regression test for issue #50825
// Make sure that the built-in bound {integer}: Sized is preferred over
// the u64: Sized bound in the where clause.

fn foo(y: &[()])
where
    u64: Sized,
{
    y[0]
}

fn main () {
    foo(&[()]);
}
