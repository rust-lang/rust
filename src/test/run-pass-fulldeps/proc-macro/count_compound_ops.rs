// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:count_compound_ops.rs
// ignore-stage1

#![feature(use_extern_macros, proc_macro_non_items)]

extern crate count_compound_ops;
use count_compound_ops::count_compound_ops;

fn main() {
    assert_eq!(count_compound_ops!(foo<=>bar <<<! -baz ++), 4);
}
