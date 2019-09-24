// run-pass
#![allow(unused_imports)]
// Regression test for issue #22246 -- we should be able to deduce
// that `&'a B::Owned` implies that `B::Owned : 'a`.

// pretty-expanded FIXME #23616

#![allow(dead_code)]

use std::ops::Deref;

pub trait ToOwned: Sized {
    type Owned: Borrow<Self>;
    fn to_owned(&self) -> Self::Owned;
}

pub trait Borrow<Borrowed> {
    fn borrow(&self) -> &Borrowed;
}

pub struct Foo<B:ToOwned> {
    owned: B::Owned
}

fn foo<B:ToOwned>(this: &Foo<B>) -> &B {
    this.owned.borrow()
}

fn main() { }
