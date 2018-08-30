// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

// Check that the destructors of simple enums are run on unwinding

use std::sync::atomic::{Ordering, AtomicUsize};
use std::thread;

static LOG: AtomicUsize = AtomicUsize::new(0);

enum WithDtor { Val }
impl Drop for WithDtor {
    fn drop(&mut self) {
        LOG.store(LOG.load(Ordering::SeqCst)+1,Ordering::SeqCst);
    }
}

pub fn main() {
    thread::spawn(move|| {
        let _e: WithDtor = WithDtor::Val;
        panic!("fail");
    }).join().unwrap_err();

    assert_eq!(LOG.load(Ordering::SeqCst), 1);
}
