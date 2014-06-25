// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty FIXME(#14193)

#![feature(managed_boxes)]

// Test cyclic detector when using trait instances.

use std::cell::RefCell;
use std::gc::{GC, Gc};

struct Tree(Gc<RefCell<TreeR>>);
struct TreeR {
    left: Option<Tree>,
    right: Option<Tree>,
    val: Box<to_str+Send>
}

trait to_str {
    fn to_str_(&self) -> String;
}

impl<T:to_str> to_str for Option<T> {
    fn to_str_(&self) -> String {
        match *self {
          None => { "none".to_string() }
          Some(ref t) => format!("some({})", t.to_str_()),
        }
    }
}

impl to_str for int {
    fn to_str_(&self) -> String {
        self.to_str()
    }
}

impl to_str for Tree {
    fn to_str_(&self) -> String {
        let Tree(t) = *self;
        let this = t.borrow();
        let (l, r) = (this.left, this.right);
        let val = &this.val;
        format!("[{}, {}, {}]", val.to_str_(), l.to_str_(), r.to_str_())
    }
}

fn foo<T:to_str>(x: T) -> String { x.to_str_() }

pub fn main() {
    let t1 = Tree(box(GC) RefCell::new(TreeR{left: None,
                                       right: None,
                                       val: box 1i as Box<to_str+Send>}));
    let t2 = Tree(box(GC) RefCell::new(TreeR{left: Some(t1),
                                       right: Some(t1),
                                       val: box 2i as Box<to_str+Send>}));
    let expected =
        "[2, some([1, none, none]), some([1, none, none])]".to_string();
    assert!(t2.to_str_() == expected);
    assert!(foo(t2) == expected);

    {
        let Tree(t1_) = t1;
        let mut t1 = t1_.borrow_mut();
        t1.left = Some(t2); // create cycle
    }
}
