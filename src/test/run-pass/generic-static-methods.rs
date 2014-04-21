// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait vec_utils<T> {
    fn map_<U>(x: &Self, f: |&T| -> U) -> Vec<U> ;
}

impl<T> vec_utils<T> for Vec<T> {
    fn map_<U>(x: &Vec<T> , f: |&T| -> U) -> Vec<U> {
        let mut r = Vec::new();
        for elt in x.iter() {
            r.push(f(elt));
        }
        r
    }
}

pub fn main() {
    assert_eq!(vec_utils::map_(&vec!(1i,2i,3i), |&x| x+1), vec!(2i,3i,4i));
}
