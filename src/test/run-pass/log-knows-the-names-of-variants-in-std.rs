
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
  b(String),
}

fn check_log<T>(exp: String, v: T) {
    assert_eq!(exp, format_strbuf!("{:?}", v));
}

pub fn main() {
    let mut x = Some(a(22u));
    let exp = "Some(a(22u))".to_string();
    let act = format_strbuf!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);

    x = None;
    let exp = "None".to_string();
    let act = format_strbuf!("{:?}", x);
    assert_eq!(act, exp);
    check_log(exp, x);
}
