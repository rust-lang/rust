// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    remove_axis(&3, 0);
}

trait Dimension {
    fn slice(&self) -> &[usize];
}

impl Dimension for () {
    fn slice(&self) -> &[usize] { &[] }
}

impl Dimension for usize {
    fn slice(&self) -> &[usize] {
        unsafe {
            ::std::slice::from_raw_parts(self, 1)
        }
    }
}

fn remove_axis(value: &usize, axis: usize) -> () {
    let tup = ();
    let mut it = tup.slice().iter();
    for (i, _) in value.slice().iter().enumerate() {
        if i == axis {
            continue;
        }
        it.next();
    }
}
