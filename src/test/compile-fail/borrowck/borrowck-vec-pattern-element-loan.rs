// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(advanced_slice_patterns)]
#![feature(slice_patterns)]

fn a<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR does not live long enough
    let tail = match vec {
        &[_, ref tail..] => tail,
        _ => panic!("a")
    };
    tail
}

fn b<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR does not live long enough
    let init = match vec {
        &[ref init.., _] => init,
        _ => panic!("b")
    };
    init
}

fn c<'a>() -> &'a [isize] {
    let vec = vec![1, 2, 3, 4];
    let vec: &[isize] = &vec; //~ ERROR does not live long enough
    let slice = match vec {
        &[_, ref slice.., _] => slice,
        _ => panic!("c")
    };
    slice
}

fn main() {}
