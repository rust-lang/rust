// xfail-fast

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;
use extra::list;

#[deriving(Clone)]
enum foo {
  a(uint),
  b(~str),
}

fn check_log<T>(exp: ~str, v: T) {
    assert_eq!(exp, fmt!("%?", v));
}

pub fn main() {
    let x = list::from_vec(~[a(22u), b(~"hi")]);
    let exp = ~"@Cons(a(22), @Cons(b(~\"hi\"), @Nil))";
    let act = fmt!("%?", x);
    assert!(act == exp);
    check_log(exp, x);
}
