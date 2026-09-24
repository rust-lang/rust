#![expect(unused)] //~ ERROR overflow evaluating the requirement
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/155423>.

use std::mem::transmute;

trait Trait {
    type Assoc<T: Trait>;
}

struct Thing;
impl Trait for Thing {
    type Assoc<T: Trait> = T::Assoc<T>;
}

fn foo<T: Trait>() -> impl Sized {
    let value: *const <T as Trait>::Assoc<T> = panic!();
    value
}

fn main() {
    let mut x = foo::<Thing>();
    x = unsafe { transmute::<_, _>(foo::<Thing>()) }
}
