// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

#[no_std];

extern mod std;

use std::str::StrVector;
use std::vec::ImmutableVector;
use std::iterator::IteratorUtil;
use std::int;

trait to_str {
    fn to_str(&self) -> ~str;
}

impl to_str for int {
    fn to_str(&self) -> ~str { int::to_str(*self) }
}

impl<T:to_str> to_str for ~[T] {
    fn to_str(&self) -> ~str {
        fmt!("[%s]", self.iter().transform(|e| e.to_str()).collect::<~[~str]>().connect(", "))
    }
}

pub fn main() {
    assert!(1.to_str() == ~"1");
    assert!((~[2, 3, 4]).to_str() == ~"[2, 3, 4]");

    fn indirect<T:to_str>(x: T) -> ~str {
        x.to_str() + ~"!"
    }
    assert!(indirect(~[10, 20]) == ~"[10, 20]!");

    fn indirect2<T:to_str>(x: T) -> ~str {
        indirect(x)
    }
    assert!(indirect2(~[1]) == ~"[1]!");
}
