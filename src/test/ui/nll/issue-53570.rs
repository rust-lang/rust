// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #53570. Here, we need to propagate that `T: 'a`
// but in some versions of NLL we were propagating a stronger
// requirement that `T: 'static`. This arose because we actually had
// to propagate both that `T: 'a` but also `T: 'b` where `'b` is the
// higher-ranked lifetime that appears in the type of the closure
// parameter `x` -- since `'b` cannot be expressed in the caller's
// space, that got promoted th `'static`.
//
// compile-pass

#![feature(nll)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

use std::cell::{RefCell, Ref};

trait AnyVec<'a> {
}

trait GenericVec<T> {
    fn unwrap<'a, 'b>(vec: &'b AnyVec<'a>) -> &'b [T] where T: 'a;
}

struct Scratchpad<'a> {
    buffers: RefCell<Box<AnyVec<'a>>>,
}

impl<'a> Scratchpad<'a> {
    fn get<T: GenericVec<T>>(&self) -> Ref<[T]>
    where T: 'a
    {
        Ref::map(self.buffers.borrow(), |x| T::unwrap(x.as_ref()))
    }
}

fn main() { }
