// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Exercise the unused_mut attribute in some positive and negative cases

#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![deny(unused_mut)]


fn main() {
    // negative cases
    let mut a = 3is; //~ ERROR: variable does not need to be mutable
    let mut a = 2is; //~ ERROR: variable does not need to be mutable
    let mut b = 3is; //~ ERROR: variable does not need to be mutable
    let mut a = vec!(3is); //~ ERROR: variable does not need to be mutable
    let (mut a, b) = (1is, 2is); //~ ERROR: variable does not need to be mutable

    match 30is {
        mut x => {} //~ ERROR: variable does not need to be mutable
    }
    match (30is, 2is) {
      (mut x, 1) | //~ ERROR: variable does not need to be mutable
      (mut x, 2) |
      (mut x, 3) => {
      }
      _ => {}
    }

    let x = |&: mut y: isize| 10is; //~ ERROR: variable does not need to be mutable
    fn what(mut foo: isize) {} //~ ERROR: variable does not need to be mutable

    // positive cases
    let mut a = 2is;
    a = 3is;
    let mut a = Vec::new();
    a.push(3is);
    let mut a = Vec::new();
    callback(|| {
        a.push(3is);
    });
    let (mut a, b) = (1is, 2is);
    a = 34;

    match 30is {
        mut x => {
            x = 21is;
        }
    }

    match (30is, 2is) {
      (mut x, 1) |
      (mut x, 2) |
      (mut x, 3) => {
        x = 21
      }
      _ => {}
    }

    let x = |&mut: mut y: isize| y = 32is;
    fn nothing(mut foo: isize) { foo = 37is; }

    // leading underscore should avoid the warning, just like the
    // unused variable lint.
    let mut _allowed = 1is;
}

fn callback<F>(f: F) where F: FnOnce() {}

// make sure the lint attribute can be turned off
#[allow(unused_mut)]
fn foo(mut a: isize) {
    let mut a = 3is;
    let mut b = vec!(2is);
}
