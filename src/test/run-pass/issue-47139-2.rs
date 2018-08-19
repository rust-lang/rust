// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #47139:
//
// Same as issue-47139-1.rs, but the impls of dummy are in the
// opposite order. This influenced the way that coherence ran and in
// some cases caused the overflow to occur when it wouldn't otherwise.
// In an effort to make the regr test more robust, I am including both
// orderings.

#![allow(dead_code)]

pub trait Insertable {
    type Values;

    fn values(self) -> Self::Values;
}

impl<T> Insertable for Option<T>
    where
    T: Insertable,
    T::Values: Default,
{
    type Values = T::Values;

    fn values(self) -> Self::Values {
        self.map(Insertable::values).unwrap_or_default()
    }
}

impl<'a, T> Insertable for &'a Option<T>
    where
    Option<&'a T>: Insertable,
{
    type Values = <Option<&'a T> as Insertable>::Values;

    fn values(self) -> Self::Values {
        self.as_ref().values()
    }
}

impl<'a, T> Insertable for &'a [T]
{
    type Values = Self;

    fn values(self) -> Self::Values {
        self
    }
}

trait Unimplemented { }

trait Dummy { }

struct Foo<T> { t: T }

impl<T> Dummy for T
    where T: Unimplemented
{ }

impl<'a, U> Dummy for Foo<&'a U>
    where &'a U: Insertable
{
}

fn main() {
}
