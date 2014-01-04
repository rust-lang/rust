// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

use std::cell::RefCell;
use std::ptr;

enum maybe_pointy {
    none,
    p(@RefCell<Pointy>),
}

struct Pointy {
    a : maybe_pointy,
    d : proc() -> uint,
}

fn make_uniq_closure<A:Send>(a: A) -> proc() -> uint {
    let result: proc() -> uint = proc() ptr::to_unsafe_ptr(&a) as uint;
    result
}

fn empty_pointy() -> @RefCell<Pointy> {
    return @RefCell::new(Pointy {
        a : none,
        d : make_uniq_closure(~"hi")
    })
}

pub fn main() {
    let v = empty_pointy();
    {
        let mut vb = v.borrow_mut();
        vb.get().a = p(v);
    }
}
