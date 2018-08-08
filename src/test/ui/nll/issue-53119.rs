// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(nll)]

use std::ops::Deref;

pub struct TypeFieldIterator<'a, T: 'a> {
    _t: &'a T,
}

pub struct Type<Id, T> {
    _types: Vec<(Id, T)>,
}

impl<'a, Id: 'a, T> Iterator for TypeFieldIterator<'a, T>
where T: Deref<Target = Type<Id, T>> {
    type Item = &'a (Id, T);

    fn next(&mut self) -> Option<&'a (Id, T)> {
        || self.next();
        None
    }
}

fn main() { }
