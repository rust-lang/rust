// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Eq;

fn sendable() {

    fn f<T:Send + Eq>(i: T, j: T) {
        assert!(i == j);
    }

    fn g<T:Send + Eq>(i: T, j: T) {
        assert!(i != j);
    }

    let i = box 100;
    let j = box 100;
    f(i, j);
    let i = box 100;
    let j = box 101;
    g(i, j);
}

fn copyable() {

    fn f<T:Eq>(i: T, j: T) {
        assert!(i == j);
    }

    fn g<T:Eq>(i: T, j: T) {
        assert!(i != j);
    }

    let i = box 100;
    let j = box 100;
    f(i, j);
    let i = box 100;
    let j = box 101;
    g(i, j);
}

fn noncopyable() {

    fn f<T:Eq>(i: T, j: T) {
        assert!(i == j);
    }

    fn g<T:Eq>(i: T, j: T) {
        assert!(i != j);
    }

    let i = box 100;
    let j = box 100;
    f(i, j);
    let i = box 100;
    let j = box 101;
    g(i, j);
}

pub fn main() {
    sendable();
    copyable();
    noncopyable();
}
