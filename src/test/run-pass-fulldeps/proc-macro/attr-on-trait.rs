// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attr-on-trait.rs
// ignore-stage1

#![feature(proc_macro, proc_macro_path_invoc)]

extern crate attr_on_trait;

trait Foo {
    #[attr_on_trait::foo]
    fn foo() {}
}

impl Foo for i32 {
    fn foo(&self) {}
}

fn main() {
    3i32.foo();
}
