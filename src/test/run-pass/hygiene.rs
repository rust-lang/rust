// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

fn main() {
    let x = 0;
    macro_rules! foo { () => {
        assert_eq!(x, 0);
    } }

    let x = 1;
    foo!();

    g();
}

fn g() {
    let x = 0;
    macro_rules! m { ($x:ident) => {
        macro_rules! m2 { () => { ($x, x) } }
        let x = 1;
        macro_rules! m3 { () => { ($x, x) } }
    } }

    let x = 2;
    m!(x);

    let x = 3;
    assert_eq!(m2!(), (2, 0));
    assert_eq!(m3!(), (2, 1));

    let x = 4;
    m!(x);
    assert_eq!(m2!(), (4, 0));
    assert_eq!(m3!(), (4, 1));
}
