// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Arg<A> {
    fn arg(&self) -> A;
}

pub trait Traversal {
    type Item;
    fn foreach<F: Arg<Self::Item>>(F);
}

impl<'a> Traversal for i32 {
    type Item = &'a i32;
    fn foreach<F: Arg<&'a i32>>(f: F) {
        f.arg();
    }
}

impl<'a> Traversal for u8 {
    type Item = &'a u8;
    // A more verbose way to refer to the associated type. Should also work
    // nonetheless.
    fn foreach<F: Arg<<Self as Traversal>::Item>>(f: F) {
        let _ = f.arg();
    }
}

fn main() {}
