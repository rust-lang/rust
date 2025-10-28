//@ run-pass

// regression test for #147964:
//   constification of these traits resulted in inference errors due to additional where clauses

use std::borrow::{Cow, Borrow};

// This won't work even with the non-const bound on `Deref` like 1.90.0, although note
// as well that no such bound existed until 1.57 started `~const` experiments.
//
// pub fn generic_deref<'a, T: ToOwned<Owned = U>, U>(cow: Cow<'a, T>) {
//     let _: &T = &cow;
// }

pub fn generic_borrow<'a, T: ToOwned<Owned = U>, U>(cow: Cow<'a, T>) {
    let _: &T = cow.borrow();
}

pub fn generic_as_ref<'a, T: ToOwned<Owned = U>, U>(cow: Cow<'a, T>) {
    let _: &T = cow.as_ref();
}

fn main() {}
