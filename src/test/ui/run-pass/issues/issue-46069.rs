// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::{Fuse, Cloned};
use std::slice::Iter;

struct Foo<'a, T: 'a>(&'a T);
impl<'a, T: 'a> Copy for Foo<'a, T> {}
impl<'a, T: 'a> Clone for Foo<'a, T> {
    fn clone(&self) -> Self { *self }
}

fn copy_ex() {
    let s = 2;
    let k1 = || s;
    let upvar = Foo(&k1);
    let k = || upvar;
    k();
}

fn main() {
    let _f = 0 as *mut <Fuse<Cloned<Iter<u8>>> as Iterator>::Item;

    copy_ex();
}
