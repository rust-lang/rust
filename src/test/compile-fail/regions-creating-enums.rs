// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum ast<'a> {
    num(usize),
    add(&'a ast<'a>, &'a ast<'a>)
}

fn build() {
    let x = ast::num(3_usize);
    let y = ast::num(4_usize);
    let z = ast::add(&x, &y);
    compute(&z);
}

fn compute(x: &ast) -> usize {
    match *x {
      ast::num(x) => { x }
      ast::add(x, y) => { compute(x) + compute(y) }
    }
}

fn map_nums<'a,'b, F>(x: &ast, f: &mut F) -> &'a ast<'b> where F: FnMut(usize) -> usize {
    match *x {
      ast::num(x) => {
        return &ast::num((*f)(x)); //~ ERROR borrowed value does not live long enough
      }
      ast::add(x, y) => {
        let m_x = map_nums(x, f);
        let m_y = map_nums(y, f);
        return &ast::add(m_x, m_y);  //~ ERROR borrowed value does not live long enough
      }
    }
}

fn main() {}
