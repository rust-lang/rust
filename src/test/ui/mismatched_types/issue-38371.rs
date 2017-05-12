// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(slice_patterns)]


struct Foo {
}

fn foo(&foo: Foo) {
}

fn bar(foo: Foo) {
}

fn qux(foo: &Foo) {
}

fn zar(&foo: &Foo) {
}

fn agh(&&bar: &u32) {
}

fn bgh(&&bar: u32) {
}

fn ugh(&[bar]: &u32) {
}

fn main() {}
