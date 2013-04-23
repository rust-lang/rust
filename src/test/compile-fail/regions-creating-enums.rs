// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum ast<'self> {
    num(uint),
    add(&'self ast<'self>, &'self ast<'self>)
}

fn build() {
    let x = num(3u);
    let y = num(4u);
    let z = add(&x, &y);
    compute(&z);
}

fn compute(x: &ast) -> uint {
    match *x {
      num(x) => { x }
      add(x, y) => { compute(x) + compute(y) }
    }
}

fn map_nums(x: &ast, f: &fn(uint) -> uint) -> &ast {
    match *x {
      num(x) => {
        return &num(f(x)); //~ ERROR illegal borrow
      }
      add(x, y) => {
        let m_x = map_nums(x, f);
        let m_y = map_nums(y, f);
        return &add(m_x, m_y);  //~ ERROR illegal borrow
      }
    }
}

fn main() {}
