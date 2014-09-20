// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1
// ignore-android
// aux-build:issue_16723_multiple_items_syntax_ext.rs
#![feature(phase)]

#[phase(plugin)] extern crate issue_16723_multiple_items_syntax_ext;

multiple_items!()

impl Struct1 {
    fn foo() {}
}
impl Struct2 {
    fn foo() {}
}

fn main() {
    Struct1::foo();
    Struct2::foo();
    println!("hallo");
}
