// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![allow(dead_code)]
#![feature(const_fn)]

// check dtor calling order when casting enums.

use std::sync::atomic;
use std::sync::atomic::Ordering;
use std::mem;

enum E {
    A = 0,
    B = 1,
    C = 2
}

static FLAG: atomic::AtomicUsize = atomic::AtomicUsize::new(0);

impl Drop for E {
    fn drop(&mut self) {
        // avoid dtor loop
        unsafe { mem::forget(mem::replace(self, E::B)) };

        FLAG.store(FLAG.load(Ordering::SeqCst)+1, Ordering::SeqCst);
    }
}

fn main() {
    assert_eq!(FLAG.load(Ordering::SeqCst), 0);
    {
        let e = E::C;
        assert_eq!(e as u32, 2);
        assert_eq!(FLAG.load(Ordering::SeqCst), 0);
    }
    assert_eq!(FLAG.load(Ordering::SeqCst), 0);
}
