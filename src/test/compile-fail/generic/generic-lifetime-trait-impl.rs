// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This code used to produce an ICE on the definition of trait Bar
// with the following message:
//
// Type parameter out of range when substituting in region 'a (root
// type=fn(Self) -> 'astr) (space=FnSpace, index=0)
//
// Regression test for issue #16218.

trait Bar<'a> {
    fn dummy(&'a self);
}

trait Foo<'a> {
    fn dummy(&'a self) { }
    fn bar<'b, T: Bar<'b>>(self) -> &'b str;
}

impl<'a> Foo<'a> for &'a str {
    fn bar<T: Bar<'a>>(self) -> &'a str { panic!() } //~ ERROR lifetime
}

fn main() {
}
