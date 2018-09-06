// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};

static A: AtomicUsize = ATOMIC_USIZE_INIT;

struct B;

impl Drop for B {
    fn drop(&mut self) {
        A.fetch_add(1, Ordering::SeqCst);
    }
}


fn test() -> bool { true }
fn test2() -> bool { false }

fn main() {
    t1();
    t2();
}

fn t1() {
    let mut a = || {
        let b = B;
        if test() {
            drop(b);
        }
        yield;
    };

    let n = A.load(Ordering::SeqCst);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
}

fn t2() {
    let mut a = || {
        let b = B;
        if test2() {
            drop(b);
        }
        yield;
    };

    let n = A.load(Ordering::SeqCst);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
}
