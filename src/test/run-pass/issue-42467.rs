// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<T>(T);

struct IntoIter<T>(T);

impl<'a, T: 'a> Iterator for IntoIter<T> {
    type Item = ();

    fn next(&mut self) -> Option<()> {
        None
    }
}

impl<T> IntoIterator for Foo<T> {
    type Item = ();
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter(self.0)
    }
}

fn main() {}
