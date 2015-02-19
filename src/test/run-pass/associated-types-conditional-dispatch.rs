// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we evaluate projection predicates to winnow out
// candidates during trait selection and method resolution (#20296).
// If we don't properly winnow out candidates based on the output type
// `Target=[A]`, then the impl marked with `(*)` is seen to conflict
// with all the others.

use std::marker::PhantomData;
use std::ops::Deref;

pub trait MyEq<U: ?Sized=Self> {
    fn eq(&self, u: &U) -> bool;
}

impl<A, B> MyEq<[B]> for [A]
    where A : MyEq<B>
{
    fn eq(&self, other: &[B]) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter())
                       .all(|(a, b)| MyEq::eq(a, b))
    }
}

// (*) This impl conflicts with everything unless the `Target=[A]`
// constraint is considered.
impl<'a, A, B, Lhs> MyEq<[B; 0]> for Lhs
    where A: MyEq<B>, Lhs: Deref<Target=[A]>
{
    fn eq(&self, other: &[B; 0]) -> bool {
        MyEq::eq(&**self, other)
    }
}

struct DerefWithHelper<H, T> {
    pub helper: H,
    pub marker: PhantomData<T>,
}

trait Helper<T> {
    fn helper_borrow(&self) -> &T;
}

impl<T> Helper<T> for Option<T> {
    fn helper_borrow(&self) -> &T {
        self.as_ref().unwrap()
    }
}

impl<T, H: Helper<T>> Deref for DerefWithHelper<H, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.helper.helper_borrow()
    }
}

pub fn check<T: MyEq>(x: T, y: T) -> bool {
    let d: DerefWithHelper<Option<T>, T> = DerefWithHelper { helper: Some(x),
                                                             marker: PhantomData };
    d.eq(&y)
}

pub fn main() {
}
