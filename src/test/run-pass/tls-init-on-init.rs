// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(thread_local_state)]

use std::thread::{self, LocalKeyState};
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

struct Foo { cnt: usize }

thread_local!(static FOO: Foo = Foo::init());

static CNT: AtomicUsize = ATOMIC_USIZE_INIT;

impl Foo {
    fn init() -> Foo {
        let cnt = CNT.fetch_add(1, Ordering::SeqCst);
        if cnt == 0 {
            FOO.with(|_| {});
        }
        Foo { cnt: cnt }
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        if self.cnt == 1 {
            FOO.with(|foo| assert_eq!(foo.cnt, 0));
        } else {
            assert_eq!(self.cnt, 0);
            match FOO.state() {
                LocalKeyState::Valid => panic!("should not be in valid state"),
                LocalKeyState::Uninitialized |
                LocalKeyState::Destroyed => {}
            }
        }
    }
}

fn main() {
    thread::spawn(|| {
        FOO.with(|_| {});
    }).join().unwrap();
}
