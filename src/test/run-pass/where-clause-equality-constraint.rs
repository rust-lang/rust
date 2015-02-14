// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(equality_constraints)]

trait Foo<A> where A = isize {
    type Inner;

    fn foo_bar(&self) where Self::Inner = String {}
}

impl Foo<isize> for () {
    type Inner = String;

    fn foo_bar(&self) {}
}

fn main() {
    ().foo_bar()
}
