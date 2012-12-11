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

enum Tree = TreeR;
type TreeR = @{
    mut left: Option<Tree>,
    mut right: Option<Tree>,
    val: to_str
};

trait to_str {
    fn to_str() -> ~str;
}

impl <T: to_str> Option<T>: to_str {
    fn to_str() -> ~str {
        match self {
          None => { ~"none" }
          Some(ref t) => { ~"some(" + t.to_str() + ~")" }
        }
    }
}

impl int: to_str {
    fn to_str() -> ~str { int::str(self) }
}

impl Tree: to_str {
    fn to_str() -> ~str {
        let l = self.left, r = self.right;
        fmt!("[%s, %s, %s]", self.val.to_str(),
             l.to_str(), r.to_str())
    }
}

fn foo<T: to_str>(x: T) -> ~str { x.to_str() }

fn main() {
    let t1 = Tree(@{mut left: None,
                    mut right: None,
                    val: 1 as to_str });
    let t2 = Tree(@{mut left: Some(t1),
                    mut right: Some(t1),
                    val: 2 as to_str });
    let expected = ~"[2, some([1, none, none]), some([1, none, none])]";
    assert t2.to_str() == expected;
    assert foo(t2 as to_str) == expected;
    t1.left = Some(t2); // create cycle
}
