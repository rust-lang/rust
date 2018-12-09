// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(clippy::many_single_char_names)]

#[derive(Copy, Clone)]
struct Foo(u8);

#[derive(Copy, Clone)]
struct Bar(u32);

fn good(a: &mut u32, b: u32, c: &Bar, d: &u32) {}

fn bad(x: &u16, y: &Foo) {}

fn main() {
    let (mut a, b, c, d, x, y) = (0, 0, Bar(0), 0, 0, Foo(0));
    good(&mut a, b, &c, &d);
    bad(&x, &y);
}
