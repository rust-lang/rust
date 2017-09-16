// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Trait<'a> {
    type T;
    type U;
    fn foo(&self, s: &'a ()) -> &'a ();
}

impl<'a> Trait<'a> for () {
    type T = &'a ();
    type U = Self::T;

    fn foo(&self, s: &'a ()) -> &'a () {
        let t: Self::T = s; t
    }
}

fn main() {}
