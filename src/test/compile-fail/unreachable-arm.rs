// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:unreachable pattern

#![feature(box_patterns)]
#![feature(box_syntax)]

enum foo { a(Box<foo>, isize), b(usize), }

fn main() { match foo::b(1_usize) { foo::b(_) | foo::a(box _, 1) => { } foo::a(_, 1) => { } } }
