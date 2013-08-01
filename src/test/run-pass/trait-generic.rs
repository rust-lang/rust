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

use std::int;

trait to_str {
    fn to_str(&self) -> ~str;
}
impl to_str for int {
    fn to_str(&self) -> ~str { int::to_str(*self) }
}
impl to_str for ~str {
    fn to_str(&self) -> ~str { self.clone() }
}
impl to_str for () {
    fn to_str(&self) -> ~str { ~"()" }
}

trait map<T> {
    fn map<U>(&self, f: &fn(&T) -> U) -> ~[U];
}
impl<T> map<T> for ~[T] {
    fn map<U>(&self, f: &fn(&T) -> U) -> ~[U] {
        let mut r = ~[];
        // FIXME: #7355 generates bad code with VecIterator
        foreach i in range(0u, self.len()) {
            r.push(f(&self[i]));
        }
        r
    }
}

fn foo<U, T: map<U>>(x: T) -> ~[~str] {
    x.map(|_e| ~"hi" )
}
fn bar<U:to_str,T:map<U>>(x: T) -> ~[~str] {
    x.map(|_e| _e.to_str() )
}

pub fn main() {
    assert_eq!(foo(~[1]), ~[~"hi"]);
    assert_eq!(bar::<int, ~[int]>(~[4, 5]), ~[~"4", ~"5"]);
    assert_eq!(bar::<~str, ~[~str]>(~[~"x", ~"y"]), ~[~"x", ~"y"]);
    assert_eq!(bar::<(), ~[()]>(~[()]), ~[~"()"]);
}
