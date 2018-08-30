// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
