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

#[deriving(Clone)]
enum foo {
  a(uint),
  b(~str),
}

fn check_log<T>(exp: ~str, v: T) {
    assert_eq!(exp, format!("{:?}", v));
}

pub fn main() {
    let mut x = Some(a(22u));
    let exp = ~"Some(a(22u))";
    let act = format!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);

    x = Some(b(~"hi"));
    let exp = ~"Some(b(~\"hi\"))";
    let act = format!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);

    x = None;
    let exp = ~"None";
    let act = format!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);
}
