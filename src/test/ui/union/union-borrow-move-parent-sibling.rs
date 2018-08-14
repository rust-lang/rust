// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(untagged_unions)]
#![allow(unused)]

#[allow(unions_with_drop_fields)]
union U {
    x: ((Vec<u8>, Vec<u8>), Vec<u8>),
    y: Box<Vec<u8>>,
}

unsafe fn parent_sibling_borrow() {
    let mut u = U { x: ((Vec::new(), Vec::new()), Vec::new()) };
    let a = &mut u.x.0;
    let a = &u.y; //~ ERROR cannot borrow `u.y`
}

unsafe fn parent_sibling_move() {
    let u = U { x: ((Vec::new(), Vec::new()), Vec::new()) };
    let a = u.x.0;
    let a = u.y; //~ ERROR use of moved value: `u.y`
}

unsafe fn grandparent_sibling_borrow() {
    let mut u = U { x: ((Vec::new(), Vec::new()), Vec::new()) };
    let a = &mut (u.x.0).0;
    let a = &u.y; //~ ERROR cannot borrow `u.y`
}

unsafe fn grandparent_sibling_move() {
    let u = U { x: ((Vec::new(), Vec::new()), Vec::new()) };
    let a = (u.x.0).0;
    let a = u.y; //~ ERROR use of moved value: `u.y`
}

unsafe fn deref_sibling_borrow() {
    let mut u = U { y: Box::default() };
    let a = &mut *u.y;
    let a = &u.x; //~ ERROR cannot borrow `u` (via `u.x`)
}

unsafe fn deref_sibling_move() {
    let u = U { x: ((Vec::new(), Vec::new()), Vec::new()) };
    let a = *u.y;
    let a = u.x; //~ ERROR use of moved value: `u.x`
}


fn main() {}
