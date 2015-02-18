// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::mem::swap;

#[derive(Debug)]
struct Ints {sum: Box<int>, values: Vec<int> }

fn add_int(x: &mut Ints, v: int) {
    *x.sum += v;
    let mut values = Vec::new();
    swap(&mut values, &mut x.values);
    values.push(v);
    swap(&mut values, &mut x.values);
}

fn iter_ints<F>(x: &Ints, mut f: F) -> bool where F: FnMut(&int) -> bool {
    let l = x.values.len();
    (0_usize..l).all(|i| f(&x.values[i]))
}

pub fn main() {
    let mut ints = box Ints {sum: box 0, values: Vec::new()};
    add_int(&mut *ints, 22);
    add_int(&mut *ints, 44);

    iter_ints(&*ints, |i| {
        println!("int = {:?}", *i);
        true
    });

    println!("ints={:?}", ints);
}
