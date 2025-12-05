//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ build-pass

// Regression test for incomplete handling of Fn-trait goals,
// fixed in #122267.

trait Trait {
    type Assoc<'a>: FnOnce(&'a ());
}

impl Trait for () {
    type Assoc<'a> = fn(&'a ());
}

trait Indir {
    fn break_me() {}
}

impl<F: Trait> Indir for F
where
    for<'a> F::Assoc<'a>: FnOnce(&'a ()),
{
    fn break_me() {}
}

fn foo<F: Trait>() {
    F::break_me()
}

fn main() {
    foo::<()>();
}
