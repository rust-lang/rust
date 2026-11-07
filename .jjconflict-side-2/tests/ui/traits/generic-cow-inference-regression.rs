//@[new] compile-flags: -Znext-solver
//@ revisions: old new
//@ run-pass

// regression test for #147964:
//   constification of these traits resulted in inference errors due to additional where clauses

use std::borrow::{Cow, Borrow};

pub fn generic_deref<'a, T: ToOwned<Owned = U>, U>(cow: Cow<'a, T>) {
    let _: &T = &cow;
}

pub fn generic_borrow<'a, T: ToOwned<Owned = U>, U>(cow: Cow<'a, T>) {
    let _: &T = cow.borrow();
}

pub fn generic_as_ref<'a, T: ToOwned<Owned = U>, U>(cow: Cow<'a, T>) {
    let _: &T = cow.as_ref();
}

fn main() {}
