// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(drop_types_in_const)]

use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};

static DROP_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

struct A(i32);

impl Drop for A {
    fn drop(&mut self) {
        // update global drop count
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
    }
}

static FOO: A = A(123);
const BAR: A = A(456);

impl A {
    const BAZ: A = A(789);
}

fn main() {
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 0);
    assert_eq!(&FOO.0, &123);
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 0);
    assert_eq!(BAR.0, 456);
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 1);
    assert_eq!(A::BAZ.0, 789);
    assert_eq!(DROP_COUNTER.load(Ordering::SeqCst), 2);
}
