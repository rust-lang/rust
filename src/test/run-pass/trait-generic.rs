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
#[legacy_modes];

trait to_str {
    fn to_str() -> ~str;
}
impl int: to_str {
    fn to_str() -> ~str { int::str(self) }
}
impl ~str: to_str {
    fn to_str() -> ~str { copy self }
}
impl (): to_str {
    fn to_str() -> ~str { ~"()" }
}

trait map<T> {
    fn map<U: Copy>(f: fn(T) -> U) -> ~[U];
}
impl<T> ~[T]: map<T> {
    fn map<U: Copy>(f: fn(T) -> U) -> ~[U] {
        let mut r = ~[];
        for self.each |x| { r += ~[f(*x)]; }
        r
    }
}

fn foo<U, T: map<U>>(x: T) -> ~[~str] {
    x.map(|_e| ~"hi" )
}
fn bar<U: to_str, T: map<U>>(x: T) -> ~[~str] {
    x.map(|_e| _e.to_str() )
}

fn main() {
    assert foo(~[1]) == ~[~"hi"];
    assert bar::<int, ~[int]>(~[4, 5]) == ~[~"4", ~"5"];
    assert bar::<~str, ~[~str]>(~[~"x", ~"y"]) == ~[~"x", ~"y"];
    assert bar::<(), ~[()]>(~[()]) == ~[~"()"];
}
