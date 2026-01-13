//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@check-pass

#![feature(try_as_dyn)]

use std::any::try_as_dyn;

trait Trait: 'static {}
trait Other {}
struct Foo<T>(T);

impl Trait for () {}
impl Trait for &'static () {}

// This impl has an implied `T: 'static` bound, but that's
// not an issue, as we just ignore all `Trait` impls where
// that would be a relevant distinguisher.
impl<T: Trait> Other for Foo<T> {}

const _: () = {
    let foo = Foo(());
    assert!(try_as_dyn::<Foo<()>, dyn Other>(&foo).is_some());
    let foo = Foo(&());
    assert!(try_as_dyn::<Foo<&'static ()>, dyn Other>(&foo).is_none());
};

fn main() {}
