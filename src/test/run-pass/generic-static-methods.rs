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
    fn map_<U>(x: &Self, f: &fn(&T) -> U) -> ~[U];
}

impl<T> vec_utils<T> for ~[T] {
    fn map_<U>(x: &~[T], f: &fn(&T) -> U) -> ~[U] {
        let mut r = ~[];
        for x.iter().advance |elt| {
            r.push(f(elt));
        }
        r
    }
}

fn main() {
    assert_eq!(vec_utils::map_(&~[1,2,3], |&x| x+1), ~[2,3,4]);
}
