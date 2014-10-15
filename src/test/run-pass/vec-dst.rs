// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn sub_expr() {
    // Test for a &[T] => &&[T] coercion in sub-expression position
    // (surpisingly, this can cause errors which are not caused by either of:
    //    `let x = vec.slice_mut(0, 2);`
    //    `foo(vec.slice_mut(0, 2));` ).
    let mut vec: Vec<int> = vec!(1, 2, 3, 4);
    let b: &mut [int] = [1, 2];
    assert!(vec.slice_mut(0, 2) == b);
}

fn index() {
    // Tests for indexing into box/& [T, ..n]
    let x: [int, ..3] = [1, 2, 3];
    let mut x: Box<[int, ..3]> = box x;
    assert!(x[0] == 1);
    assert!(x[1] == 2);
    assert!(x[2] == 3);
    x[1] = 45;
    assert!(x[0] == 1);
    assert!(x[1] == 45);
    assert!(x[2] == 3);

    let mut x: [int, ..3] = [1, 2, 3];
    let x: &mut [int, ..3] = &mut x;
    assert!(x[0] == 1);
    assert!(x[1] == 2);
    assert!(x[2] == 3);
    x[1] = 45;
    assert!(x[0] == 1);
    assert!(x[1] == 45);
    assert!(x[2] == 3);
}

pub fn main() {
    sub_expr();
    index();
}
