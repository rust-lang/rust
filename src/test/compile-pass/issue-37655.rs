// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #37655. The problem was a false edge created by
// coercion that wound up requiring that `'a` (in `split()`) outlive
// `'b`, which shouldn't be necessary.

#![allow(warnings)]

trait SliceExt<T> {
    type Item;

    fn get_me<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<Self::Item>;
}

impl<T> SliceExt<T> for [T] {
    type Item = T;

    fn get_me<I>(&self, index: I) -> &I::Output
        where I: SliceIndex<T>
    {
        panic!()
    }
}

pub trait SliceIndex<T> {
    type Output: ?Sized;
}

impl<T> SliceIndex<T> for usize {
    type Output = T;
}

fn foo<'a, 'b>(split: &'b [&'a [u8]]) -> &'a [u8] {
    split.get_me(0)
}

fn main() { }
