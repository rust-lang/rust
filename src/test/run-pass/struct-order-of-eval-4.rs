// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that struct-literal expression order-of-eval is as expected
// even when no Drop-implementations are involved.

// pretty-expanded FIXME #23616

use std::sync::atomic::{Ordering, AtomicUsize, ATOMIC_USIZE_INIT};

struct W { wrapped: u32 }
struct S { f0: W, _f1: i32 }

pub fn main() {
    const VAL: u32 = 0x89AB_CDEF;
    let w = W { wrapped: VAL };
    let s = S {
        _f1: { event(0x01); 23 },
        f0: { event(0x02); w },
    };
    assert_eq!(s.f0.wrapped, VAL);
    let actual = event_log();
    let expect = 0x01_02;
    assert!(expect == actual,
            "expect: 0x{:x} actual: 0x{:x}", expect, actual);
}

static LOG: AtomicUsize = ATOMIC_USIZE_INIT;

fn event_log() -> usize {
    LOG.load(Ordering::SeqCst)
}

fn event(tag: u8) {
    let old_log = LOG.load(Ordering::SeqCst);
    let new_log = (old_log << 8) + tag as usize;
    LOG.store(new_log, Ordering::SeqCst);
}
