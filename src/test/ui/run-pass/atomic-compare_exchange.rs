// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(extended_compare_and_swap)]
use std::sync::atomic::{AtomicIsize, ATOMIC_ISIZE_INIT};
use std::sync::atomic::Ordering::*;

static ATOMIC: AtomicIsize = ATOMIC_ISIZE_INIT;

fn main() {
    // Make sure codegen can emit all the intrinsics correctly
    ATOMIC.compare_exchange(0, 1, Relaxed, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Release, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, SeqCst).ok();
    ATOMIC.compare_exchange_weak(0, 1, Relaxed, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Release, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, SeqCst).ok();
}
