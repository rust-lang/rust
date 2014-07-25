// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum BogusOption<T> {
    None,
    Some(T),
}

type Iterator = int;

pub fn main() {
    let x = [ 3i, 3, 3 ];
    for i in x.iter() {
        assert_eq!(*i, 3);
    }
}

