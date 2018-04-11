// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generic_associated_types)]

use std::ops::Deref;

//FIXME(#44265): "lifetime parameters are not allowed on this type" errors will be addressed in a
//follow-up PR

trait Iterable {
    type Item<'a>;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>;
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
}

// Impl for struct type
impl<T> Iterable for Vec<T> {
    type Item<'a> = &'a T;
    type Iter<'a> = std::slice::Iter<'a, T>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
        self.iter()
    }
}

// Impl for a primitive type
impl<T> Iterable for [T] {
    type Item<'a> = &'a T;
    type Iter<'a> = std::slice::Iter<'a, T>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
        self.iter()
    }
}

fn make_iter<'a, I: Iterable>(it: &'a I) -> I::Iter<'a> {
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
    it.iter()
}

fn get_first<'a, I: Iterable>(it: &'a I) -> Option<I::Item<'a>> {
    //~^ ERROR lifetime parameters are not allowed on this type [E0110]
    it.iter().next()
}

fn main() {}
