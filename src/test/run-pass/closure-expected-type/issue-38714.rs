// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct UsizeRef<'a> {
    a: &'a usize
}

type RefTo = Box<for<'r> Fn(&'r Vec<usize>) -> UsizeRef<'r>>;

fn ref_to<'a>(vec: &'a Vec<usize>) -> UsizeRef<'a> {
    UsizeRef{ a: &vec[0]}
}

fn main() {
    // Regression test: this was causing ICEs; it should compile.
    let a: RefTo = Box::new(|vec: &Vec<usize>| {
        UsizeRef{ a: &vec[0] }
    });
}
