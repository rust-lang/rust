//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] dont-require-annotations: ERROR
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #151069

// Regression test for #151069. This triggers a caching bug in the old solver,
// resulting in an ICE. IT results in a bunch of overflow errors, so let's just
// ignore the annotations here.

trait Trait {
    type Assoc2;
}
struct Bar;
impl Trait for Bar
where
    <Bar as Trait>::Assoc2: Trait,
{
    type Assoc2 = ();
}
struct Foo {
    field: <Bar as Trait>::Assoc2,
}
static FOO2: &Foo = 0;
fn main() {}
