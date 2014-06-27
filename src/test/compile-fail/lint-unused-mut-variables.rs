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

#![allow(dead_assignment)]
#![allow(unused_variable)]
#![allow(dead_code)]
#![deny(unused_mut)]


fn main() {
    // negative cases
    let mut a = 3i; //~ ERROR: variable does not need to be mutable
    let mut a = 2i; //~ ERROR: variable does not need to be mutable
    let mut b = 3i; //~ ERROR: variable does not need to be mutable
    let mut a = vec!(3i); //~ ERROR: variable does not need to be mutable
    let (mut a, b) = (1i, 2i); //~ ERROR: variable does not need to be mutable

    match 30i {
        mut x => {} //~ ERROR: variable does not need to be mutable
    }
    match (30i, 2i) {
      (mut x, 1) | //~ ERROR: variable does not need to be mutable
      (mut x, 2) |
      (mut x, 3) => {
      }
      _ => {}
    }

    let x = |mut y: int| 10i; //~ ERROR: variable does not need to be mutable
    fn what(mut foo: int) {} //~ ERROR: variable does not need to be mutable

    // positive cases
    let mut a = 2i;
    a = 3i;
    let mut a = Vec::new();
    a.push(3i);
    let mut a = Vec::new();
    callback(|| {
        a.push(3i);
    });
    let (mut a, b) = (1i, 2i);
    a = 34;

    match 30i {
        mut x => {
            x = 21i;
        }
    }

    match (30i, 2i) {
      (mut x, 1) |
      (mut x, 2) |
      (mut x, 3) => {
        x = 21
      }
      _ => {}
    }

    let x = |mut y: int| y = 32i;
    fn nothing(mut foo: int) { foo = 37i; }

    // leading underscore should avoid the warning, just like the
    // unused variable lint.
    let mut _allowed = 1i;
}

fn callback(f: ||) {}

// make sure the lint attribute can be turned off
#[allow(unused_mut)]
fn foo(mut a: int) {
    let mut a = 3i;
    let mut b = vec!(2i);
}
