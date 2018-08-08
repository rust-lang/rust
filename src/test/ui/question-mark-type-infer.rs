// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(question_mark, question_mark_carrier)]

// Test that type inference fails where there are multiple possible return types
// for the `?` operator.

fn f(x: &i32) -> Result<i32, ()> {
    Ok(*x)
}

fn g() -> Result<Vec<i32>, ()> {
    let l = [1, 2, 3, 4];
    l.iter().map(f).collect()? //~ ERROR type annotations required: cannot resolve
}

fn main() {
    g();
}
