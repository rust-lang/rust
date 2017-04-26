// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo() {}
fn bar(x: &usize) {}
fn qux(x: &usize, y: &usize) -> std::cmp::Ordering { panic!() }

fn main() {
    let mut vec = [1, 2, 3];
    vec.sort_by(|| panic!());
    vec.sort_by(|tuple| panic!());
    vec.sort_by(|(tuple, tuple2)| panic!());
    vec.sort_by(foo);
    vec.sort_by(bar);
    vec.sort_by(qux);
}
