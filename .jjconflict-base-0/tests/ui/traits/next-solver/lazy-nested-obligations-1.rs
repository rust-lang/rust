//@ check-pass
//@ compile-flags: -Znext-solver
// Issue 94358

fn foo<C>(_: C)
where
    for <'a> &'a C: IntoIterator,
    for <'a> <&'a C as IntoIterator>::IntoIter: ExactSizeIterator,
{}

fn main() {
    foo::<_>(vec![true, false]);
}
