// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait LineFormatter<'a> {
    type Iter: Iterator<Item=&'a str> + 'a;
    fn iter(&'a self, line: &'a str) -> Self::Iter;

    fn dimensions(&'a self, line: &'a str) {
        let iter: Self::Iter = self.iter(line);
        <_ as IntoIterator>::into_iter(iter);
    }
}

fn main() {}
