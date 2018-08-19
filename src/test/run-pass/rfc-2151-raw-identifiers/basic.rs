// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(raw_identifiers)]

fn r#fn(r#match: u32) -> u32 {
    r#match
}

pub fn main() {
    let r#struct = 1;
    assert_eq!(1, r#struct);

    let foo = 2;
    assert_eq!(2, r#foo);

    let r#bar = 3;
    assert_eq!(3, bar);

    assert_eq!(4, r#fn(4));

    let r#true = false;
    assert_eq!(r#true, false);
}
