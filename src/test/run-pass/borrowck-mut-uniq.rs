// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::swap;

struct Ints {sum: ~int, values: ~[int]}

fn add_int(x: &mut Ints, v: int) {
    *x.sum += v;
    let mut values = ~[];
    swap(&mut values, &mut x.values);
    values.push(v);
    swap(&mut values, &mut x.values);
}

fn iter_ints(x: &Ints, f: |x: &int| -> bool) -> bool {
    let l = x.values.len();
    range(0u, l).advance(|i| f(&x.values[i]))
}

pub fn main() {
    let mut ints = ~Ints {sum: ~0, values: ~[]};
    add_int(ints, 22);
    add_int(ints, 44);

    iter_ints(ints, |i| {
        println!("int = {}", *i);
        true
    });

    println!("ints={:?}", ints);
}
