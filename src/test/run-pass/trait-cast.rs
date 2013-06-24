// xfail-test FIXME #5882
// Weird borrow check bug

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test cyclic detector when using trait instances.

struct Tree(@mut TreeR);
struct TreeR {
    left: Option<Tree>,
    right: Option<Tree>,
    val: ~to_str
}

trait to_str {
    fn to_str_(&self) -> ~str;
}

impl<T:to_str> to_str for Option<T> {
    fn to_str_(&self) -> ~str {
        match *self {
          None => { ~"none" }
          Some(ref t) => { ~"some(" + t.to_str_() + ~")" }
        }
    }
}

impl to_str for int {
    fn to_str_(&self) -> ~str { self.to_str() }
}

impl to_str for Tree {
    fn to_str_(&self) -> ~str {
        let (l, r) = (self.left, self.right);
        let val = &self.val;
        fmt!("[%s, %s, %s]", val.to_str_(), l.to_str_(), r.to_str_())
    }
}

fn foo<T:to_str>(x: T) -> ~str { x.to_str_() }

pub fn main() {
    let t1 = Tree(@mut TreeR{left: None,
                             right: None,
                             val: ~1 as ~to_str });
    let t2 = Tree(@mut TreeR{left: Some(t1),
                             right: Some(t1),
                             val: ~2 as ~to_str });
    let expected = ~"[2, some([1, none, none]), some([1, none, none])]";
    assert!(t2.to_str_() == expected);
    assert!(foo(t2) == expected);
    t1.left = Some(t2); // create cycle
}
