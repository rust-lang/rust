// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_patterns)]
#![feature(box_syntax)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(unreachable_patterns)]

enum IntList {
    Cons(isize, Box<IntList>),
    Nil
}

fn tail(source_list: &IntList) -> IntList {
    match source_list {
        &IntList::Cons(val, box ref next_list) => tail(next_list),
        &IntList::Cons(val, box IntList::Nil)  => IntList::Cons(val, box IntList::Nil),
//~^ ERROR unreachable pattern
        _                          => panic!()
    }
}

fn main() {}
