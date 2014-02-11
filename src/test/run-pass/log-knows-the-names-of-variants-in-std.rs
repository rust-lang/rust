// ignore-fast

// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod collections;
use collections::list;

#[deriving(Clone)]
enum foo {
  a(uint),
  b(~str),
}

fn check_log<T>(exp: ~str, v: T) {
    assert_eq!(exp, format!("{:?}", v));
}

pub fn main() {
    let x = list::from_vec([a(22u), b(~"hi")]);
    let exp = ~"@Cons(a(22u), @Cons(b(~\"hi\"), @Nil))";
    let act = format!("{:?}", x);
    assert!(act == exp);
    check_log(exp, x);
}
