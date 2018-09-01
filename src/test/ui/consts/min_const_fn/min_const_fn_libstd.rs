// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(integer_atomics, min_const_fn)]

// compile-pass

use std::cell::UnsafeCell;
use std::sync::atomic::AtomicU32;
pub struct Condvar {
    condvar: UnsafeCell<AtomicU32>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct NoWait(u32);

const CONDVAR_HAS_NO_WAITERS: NoWait = NoWait(42);

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar {
            condvar: UnsafeCell::new(AtomicU32::new(CONDVAR_HAS_NO_WAITERS.0)),
        }
    }
}

fn main() {}
