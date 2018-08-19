// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(nll)]

use std::borrow::Cow;

#[derive(Clone, Debug)]
struct S<'a> {
    name: Cow<'a, str>
}

#[derive(Clone, Debug)]
struct T<'a> {
    s: Cow<'a, [S<'a>]>
}

fn main() {
    let s1 = [S { name: Cow::Borrowed("Test1") }, S { name: Cow::Borrowed("Test2") }];
    let b1 = T { s: Cow::Borrowed(&s1) };
    let s2 = [S { name: Cow::Borrowed("Test3") }, S { name: Cow::Borrowed("Test4") }];
    let b2 = T { s: Cow::Borrowed(&s2) };

    let mut v = Vec::new();
    v.push(b1);
    v.push(b2);

    println!("{:?}", v);
}
