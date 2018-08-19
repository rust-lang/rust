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

#![feature(thread_local_try_with)]

use std::thread;

static mut DROP_RUN: bool = false;

struct Foo;

thread_local!(static FOO: Foo = Foo {});

impl Drop for Foo {
    fn drop(&mut self) {
        assert!(FOO.try_with(|_| panic!("`try_with` closure run")).is_err());
        unsafe { DROP_RUN = true; }
    }
}

fn main() {
    thread::spawn(|| {
        assert_eq!(FOO.try_with(|_| {
            132
        }).expect("`try_with` failed"), 132);
    }).join().unwrap();
    assert!(unsafe { DROP_RUN });
}
