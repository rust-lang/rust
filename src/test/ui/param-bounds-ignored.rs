// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// must-compile-successfully

use std::rc::Rc;

type SVec<T: Send> = Vec<T>;
type VVec<'b, 'a: 'b> = Vec<&'a i32>;
type WVec<'b, T: 'b> = Vec<T>;

fn foo<'a>(y: &'a i32) {
    // If the bounds above would matter, the code below would be rejected.
    let mut x : SVec<_> = Vec::new();
    x.push(Rc::new(42));

    let mut x : VVec<'static, 'a> = Vec::new();
    x.push(y);

    let mut x : WVec<'static, & 'a i32> = Vec::new();
    x.push(y);
}

fn main() {
    foo(&42);
}
