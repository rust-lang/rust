// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::string_add)]
#[allow(clippy::string_add_assign)]
fn add_only() {
    // ignores assignment distinction
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[warn(clippy::string_add_assign)]
fn add_assign_only() {
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[warn(clippy::string_add, clippy::string_add_assign)]
fn both() {
    let mut x = "".to_owned();

    for _ in 1..3 {
        x = x + ".";
    }

    let y = "".to_owned();
    let z = y + "...";

    assert_eq!(&x, &z);
}

#[allow(dead_code, unused_variables)]
#[warn(clippy::string_lit_as_bytes)]
fn str_lit_as_bytes() {
    let bs = "hello there".as_bytes();

    let bs = r###"raw string with three ### in it and some " ""###.as_bytes();

    // no warning, because this cannot be written as a byte string literal:
    let ubs = "☃".as_bytes();

    let strify = stringify!(foobar).as_bytes();

    let includestr = include_str!("entry.rs").as_bytes();
}

#[allow(clippy::assign_op_pattern)]
fn main() {
    add_only();
    add_assign_only();
    both();

    // the add is only caught for `String`
    let mut x = 1;
    x = x + 1;
    assert_eq!(2, x);
}
