// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::cell::RefCell;
use std::gc::{Gc, GC};

enum maybe_pointy {
    none,
    p(Gc<RefCell<Pointy>>),
}

struct Pointy {
    a : maybe_pointy,
    c : Box<int>,
    d : proc():Send->(),
}

fn empty_pointy() -> Gc<RefCell<Pointy>> {
    return box(GC) RefCell::new(Pointy {
        a : none,
        c : box 22,
        d : proc() {},
    })
}

pub fn main() {
    let v = empty_pointy();
    {
        let mut vb = v.borrow_mut();
        vb.a = p(v);
    }
}
