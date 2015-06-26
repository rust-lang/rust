// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(anon_lifetime)]

struct Foo<'a, 'b: 'a, T: 'b>(&'a Vec<&'b T>);

fn from<'c, 'd, T>(foo: &'c Foo<'_, 'd, T>) -> Result<&'c T, &'d T> {
    if 1 == 1 {
        Ok(&foo.0[0])
    } else {
        Err(&foo.0[1])
    }
}

impl<'a, 'b, T> Foo<'a, 'b, T> {
    fn to(&self) {
        let _: Result<&'_ T, &'_ _> = from/*::<'_, 'b>*/(self);
    }
}

pub fn main() {}
