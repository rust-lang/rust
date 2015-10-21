// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test ensures that vec.into_iter does not overconstrain element lifetime.

pub fn main() {
    original_report();
    revision_1();
    revision_2();
}

fn original_report() {
    drop(vec![&()].into_iter())
}

fn revision_1() {
    // below is what above `vec!` expands into at time of this writing.
    drop(<[_]>::into_vec(::std::boxed::Box::new([&()])).into_iter())
}

fn revision_2() {
    drop((match (Vec::new(), &()) { (mut v, b) => { v.push(b); v } }).into_iter())
}
