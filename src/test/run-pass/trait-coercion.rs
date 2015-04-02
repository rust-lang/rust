// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax, old_io, io)]

use std::io::{self, Write};

trait Trait {
    fn f(&self);
}

#[derive(Copy, Clone)]
struct Struct {
    x: isize,
    y: isize,
}

impl Trait for Struct {
    fn f(&self) {
        println!("Hi!");
    }
}

fn foo(mut a: Box<Write>) {}

// FIXME (#22405): Replace `Box::new` with `box` here when/if possible.

pub fn main() {
    let a = Struct { x: 1, y: 2 };
    let b: Box<Trait> = Box::new(a);
    b.f();
    let c: &Trait = &a;
    c.f();

    let out = io::stdout();
    foo(Box::new(out));
}
