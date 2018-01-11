// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(never_type)]
#![feature(type_ascription)]

struct Foo {
    a: usize,
    b: usize,
}

fn a() {
    // struct expr is unreachable:
    let x = Foo { a: 22, b: 33, ..return }; //~ ERROR unreachable
}

fn b() {
    // the `33` is unreachable:
    let x = Foo { a: return, b: 33, ..return }; //~ ERROR unreachable
}

fn c() {
    // the `..return` is unreachable:
    let x = Foo { a: 22, b: return, ..return }; //~ ERROR unreachable
}

fn d() {
    // the struct expr is unreachable:
    let x = Foo { a: 22, b: return }; //~ ERROR unreachable
}

fn main() { }
