// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<T>(mut list: &mut List<T>) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.value);
        if let Some(n) = list.next.as_mut() {
            list = n;
        } else {
            return result;
        }
    }
}

fn main() {
    let mut list = List { value: 1, next: None };
    let vec = to_refs(&mut list);
    assert_eq!(vec![&mut 1], vec);
}
