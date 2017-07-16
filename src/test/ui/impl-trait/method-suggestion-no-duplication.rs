// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// issue #21405
// ignore-tidy-linelength

struct Foo;

fn foo<F>(f: F) where F: FnMut(Foo) {}

fn main() {
    foo(|s| s.is_empty());
    //~^ ERROR no method named `is_empty` found
    //~^^ HELP #1: `std::iter::ExactSizeIterator`
    //~^^^ HELP #2: `core::slice::SliceExt`
    //~^^^^ HELP #3: `core::str::StrExt`
    //~^^^^^ HELP items from traits can only be used if the trait is implemented and in scope; the following traits define an item `is_empty`, perhaps you need to implement one of them:
}
