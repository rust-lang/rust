//@ compile-flags: -Znext-solver
//@ check-pass

// Fixes a regression in `receiver_is_valid` in wfcheck where we were using
// `InferCtxt::can_eq` instead of processing alias-relate goals, leading to false
// positives, not deref'ing enough steps to check the receiver is valid.

trait Mirror {
    type Mirror: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type Mirror = T;
}

trait Foo {
    fn foo(&self) {}
}

impl Foo for <() as Mirror>::Mirror {
    fn foo(&self) {}
}

fn main() {}
