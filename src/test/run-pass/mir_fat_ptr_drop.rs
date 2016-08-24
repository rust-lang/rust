// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that ordinary fat pointer operations work.

#![feature(braced_empty_structs)]
#![feature(rustc_attrs)]

use std::sync::atomic;
use std::sync::atomic::Ordering::SeqCst;

static COUNTER: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;

struct DropMe {
}

impl Drop for DropMe {
    fn drop(&mut self) {
        COUNTER.fetch_add(1, SeqCst);
    }
}

fn fat_ptr_move_then_drop(a: Box<[DropMe]>) {
    let b = a;
}

fn main() {
    let a: Box<[DropMe]> = Box::new([DropMe { }]);
    fat_ptr_move_then_drop(a);
    assert_eq!(COUNTER.load(SeqCst), 1);
}
