// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(atomic_access)]
use std::sync::atomic::{AtomicBool, ATOMIC_BOOL_INIT};
use std::sync::atomic::Ordering::*;

static mut ATOMIC: AtomicBool = ATOMIC_BOOL_INIT;

fn main() {
    unsafe {
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.store(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_or(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_and(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.fetch_nand(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_xor(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
    }
}
