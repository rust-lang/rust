// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lifetimes.rs
// ignore-stage1

#![feature(proc_macro)]

extern crate lifetimes;
use lifetimes::*;

lifetimes_bang! {
    fn bang<'a>() -> &'a u8 { &0 }
}

#[lifetimes_attr]
fn attr<'a>() -> &'a u8 { &1 }

#[derive(Lifetimes)]
pub struct Lifetimes<'a> {
    pub field: &'a u8,
}

fn main() {
    assert_eq!(bang::<'static>(), &0);
    assert_eq!(attr::<'static>(), &1);
    let l1 = Lifetimes { field: &0 };
    let l2 = m::Lifetimes { field: &1 };
}
